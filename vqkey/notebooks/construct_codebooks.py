# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import *
from transformers.cache_utils import DynamicCache
import torch.multiprocessing as mp
import datasets
from tqdm import tqdm 
import torch
import json
import types
import uuid
import random
import string

from kmeans_gpu import KMeansPlusPlus

# %%
model_dir = "../../huggingface-models/Llama-3.1-8B-Instruct"

# %%
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, _attn_implementation="eager").to("cuda")

# %%
num_hidden_layers = model.config.num_hidden_layers
num_attention_heads = model.config.num_attention_heads
num_key_value_heads = model.config.num_key_value_heads
num_key_value_groups = num_attention_heads // num_key_value_heads
head_dim = model.config.head_dim
dtype = torch.bfloat16

# %%
target_num_tokens = 64 * 1024
target_num_random_tokens = 16 * 1024

# %%
dataset_dir = "../../huggingface-datasets/fineweb-edu"

# %%
dataset = datasets.load_dataset(dataset_dir, data_files="sample/10BT/000_00000.parquet")

# %%
def llama_attention_forward_wrapper(query_states_stored, key_states_stored):
    rerope_relative_pos = 2048

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin) if q is not None else None
        k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None
        return q_embed, k_embed

    def llama_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        device = hidden_states.device
        dtype = hidden_states.dtype

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos_rerope, sin_rerope = self.rotary_emb(value_states, torch.tensor([rerope_relative_pos], device=device, dtype=torch.long).unsqueeze(0))
        query_states_rerope, _ = apply_rotary_pos_emb(query_states, None, cos_rerope, sin_rerope)

        key_states_stored[self.layer_idx] = torch.cat((key_states_stored[self.layer_idx], key_states[0].detach().to("cpu")), dim=-2)
        query_states_stored[self.layer_idx] = torch.cat((query_states_stored[self.layer_idx], query_states_rerope[0].detach().to("cpu")), dim=-2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    return llama_attention_forward

# %%
key_states_stored = [torch.zeros((num_key_value_heads, 0, head_dim), dtype=torch.bfloat16) for _ in range(num_hidden_layers)]
query_states_stored = [torch.zeros((num_attention_heads, 0, head_dim), dtype=torch.bfloat16) for _ in range(num_hidden_layers)]

for layer in model.model.layers:
    layer.self_attn.forward = types.MethodType(llama_attention_forward_wrapper(query_states_stored, key_states_stored), layer.self_attn)

# %%
chunk_size = 8192
max_len = 16 * 1024
num_tokens = 0

with torch.no_grad():
    with tqdm(desc="[INFO] collecting key states", total=target_num_tokens) as pbar:
        for sample_idx in range(len(dataset["train"])):
            if num_tokens >= target_num_tokens:
                break
            
            input_ids = tokenizer.encode(dataset["train"][sample_idx]["text"])
            input_ids = input_ids[:max_len]

            kv_cache = DynamicCache()
            for chunk_begin in range(0, len(input_ids), chunk_size):
                chunk_end = min(chunk_begin + chunk_size, len(input_ids))
                kv_cache = model(torch.tensor(input_ids[chunk_begin:chunk_end], dtype=torch.long).unsqueeze(0).to("cuda"), past_key_values=kv_cache).past_key_values
            
            pbar.update(len(input_ids))

            num_tokens += len(input_ids)

print(f"[INFO] processed {sample_idx} samples")

# %%
num_uuid = 128 * 1024
random_string = ", ".join([str(uuid.uuid4()) for _ in range(num_uuid)])

random_string[:256]

# %%
chunk_size = 2048

with torch.no_grad():
    input_ids = tokenizer.encode(random_string)
    input_ids = input_ids[:target_num_random_tokens]

    kv_cache = DynamicCache()
    for chunk_begin in tqdm(range(0, len(input_ids), chunk_size), desc="collecting key states on random string"):
        chunk_end = min(chunk_begin + chunk_size, len(input_ids))
        kv_cache = model(torch.tensor(input_ids[chunk_begin:chunk_end], dtype=torch.long).unsqueeze(0).to("cuda"), past_key_values=kv_cache).past_key_values

# %%
epsilon = 1e-6

cholesky_factors = [torch.zeros(num_key_value_heads, head_dim, head_dim, dtype=dtype) for _ in range(num_hidden_layers)]
inv_cholesky_factors = [torch.zeros(num_key_value_heads, head_dim, head_dim, dtype=dtype) for _ in range(num_hidden_layers)]
num_key_value_groups = num_attention_heads // num_key_value_heads

for layer_idx in tqdm(range(num_hidden_layers)):
    for key_value_head_idx in range(num_key_value_heads):
        q = query_states_stored[layer_idx][num_key_value_groups*key_value_head_idx:num_key_value_groups*(key_value_head_idx+1)].reshape(-1, head_dim).to(torch.float32)
        H = q.T @ q
        H = H / (H**2).mean().sqrt()
        L = torch.linalg.cholesky(H + epsilon * torch.eye(H.size(0)))
        inv_cholesky_factors[layer_idx][key_value_head_idx] = torch.linalg.inv(L)
        cholesky_factors[layer_idx][key_value_head_idx] = L

# %%
codebook_size = 4096

codebooks = [torch.zeros((num_key_value_heads, codebook_size, head_dim)) for _ in range(num_hidden_layers)]

for layer_idx in range(num_hidden_layers):

    for key_value_head_idx in tqdm(range(model.config.num_key_value_heads), desc=f"Layer {layer_idx}"):
        kmeans = KMeansPlusPlus(n_clusters=codebook_size, device="cuda")
        kmeans.fit(key_states_stored[layer_idx][key_value_head_idx] @ cholesky_factors[layer_idx][key_value_head_idx])
        codebooks[layer_idx][key_value_head_idx] = kmeans.centroids

# %%
import os
os.makedirs('../codebooks', exist_ok=True)
torch.save((cholesky_factors, inv_cholesky_factors), "../codebooks/llama-3-8b-cholesky_factors.pt")
torch.save(codebooks, "../codebooks/llama-3-8b-codebooks.pt")


