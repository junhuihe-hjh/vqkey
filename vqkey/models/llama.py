import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import *
from transformers.cache_utils import *
import types
from torch.nn.attention import SDPBackend, sdpa_kernel

import selective_attention

class DecodingStatistics:
    def __init__(self):
        self.workload = 0.0
        self.num_decode_tokens = 0


class LLM:
    def __init__(
            self, 
            model_name: str, 
            codebook_dir: str, 
            codebook_name: str = None,
            cholesky_factors: str = None,
            device: str = "cuda", 
            dtype = torch.bfloat16,
            wrope_local_window: int = 64,
            wrope_relative_pos: int = 2048,
            attention_sink : int = 8,
            sampling_ratio: float = 0.0,
            topk: float = 0.0,
        ):
        self.device = device
        self.dtype = dtype

        self.decoding_statistics = DecodingStatistics()
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
 
        if sampling_ratio != 0.0 or topk != 0.0:
            self.use_vq_cache = True

            self.hf_model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device, _attn_implementation="eager")

            if cholesky_factors is not None:
                cholesky_factors, inv_cholesky_factors = torch.load(f"{codebook_dir}/{cholesky_factors}", weights_only=False)
            else:
                cholesky_factors, inv_cholesky_factors = None, None

            codebooks = torch.load(f"{codebook_dir}/{codebook_name}", weights_only=False)

            cos_wrope, sin_wrope = self.hf_model.model.rotary_emb(
                torch.zeros((1,), device=device, dtype=dtype),
                torch.tensor(
                    [[wrope_relative_pos]], 
                    device=device, 
                    dtype=torch.int64
                )
            )

            for layer_idx, layer in enumerate(self.hf_model.model.layers):
                layer.self_attn.codebook = codebooks[layer_idx].to(self.device).to(self.dtype)
                layer.self_attn.cholesky_factor = cholesky_factors[layer_idx].to(self.device).to(self.dtype)
                layer.self_attn.inv_cholesky_factor = inv_cholesky_factors[layer_idx].to(self.device).to(self.dtype)

                layer.self_attn.cos_wrope, layer.self_attn.sin_wrope = cos_wrope, sin_wrope

                layer.self_attn.forward = types.MethodType(
                    llama_attention_forward_wrapper(
                        wrope_local_window,
                        wrope_relative_pos,
                        attention_sink,
                        sampling_ratio,
                        topk,
                        self.decoding_statistics,
                    ), 
                    layer.self_attn
                )
        else:
            self.use_vq_cache = False

            self.hf_model = LlamaForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=dtype, 
                device_map=device, 
                _attn_implementation="sdpa"
            )
        
    def generate(self, input_ids, max_gen=128, chunk_size=2048):
        with torch.no_grad():
            prefill_len = len(input_ids)
            
            if self.use_vq_cache:
                kv_cache = VQCacheOffload(
                    config=self.hf_model.config, batch_size=1, 
                    max_cache_len=prefill_len + max_gen, 
                    dtype=self.dtype,
                    device=self.device
                )
            else:
                kv_cache = DynamicCache()

            for begin_idx in range(0, prefill_len, chunk_size):
                end_idx = min(begin_idx + chunk_size, prefill_len)
                output = self.hf_model(torch.tensor(input_ids[begin_idx:end_idx]).unsqueeze(0).to(self.device), past_key_values=kv_cache)
                kv_cache = output.past_key_values
            
            input_ids.append(output.logits[0, -1].argmax(dim=-1).item())

            for _ in range(max_gen):
                output = self.hf_model(torch.tensor(input_ids[-1:]).unsqueeze(0).to(self.device), past_key_values=kv_cache)

                kv_cache = output.past_key_values
                input_ids.append(output.logits[0, -1].argmax(dim=-1).item())

                if input_ids[-1] == self.hf_tokenizer.eos_token_id:
                    break
            
            return input_ids[prefill_len:]

    def benchmark_batched_inference(self, input_ids, bsz=1, warmup_len=8, test_len=128, chunk_size=128 * 1024):
        with torch.no_grad():
            prefill_len = len(input_ids) - warmup_len - test_len
            
            if self.use_vq_cache:
                kv_cache = VQCacheOffload(
                    config=self.hf_model.config, 
                    batch_size=1,
                    max_cache_len=len(input_ids), 
                    device=self.device, 
                    dtype=self.dtype
                )
            else:
                # kv_cache = StaticCache(
                #     config=self.hf_model.config, 
                #     batch_size=1,
                #     max_cache_len=len(input_ids), 
                #     device=self.device, 
                #     dtype=self.dtype
                # )
                kv_cache = DynamicCache()

            torch.cuda.synchronize()
            t_start = time.time()

            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                for begin_idx in tqdm(range(0, prefill_len, chunk_size), desc="prefill"):
                    end_idx = min(begin_idx + chunk_size, prefill_len)
                    output = self.hf_model(torch.tensor(input_ids[begin_idx:end_idx]).unsqueeze(0).to(self.device), past_key_values=kv_cache)
                    kv_cache = output.past_key_values

            if isinstance(kv_cache, VQCacheOffload):
                kv_cache.switch_between_cpu_and_gpu()
            
            torch.cuda.synchronize()
            t_end = time.time()
            print(f"[INFO] prefilling latency: {t_end - t_start}s")

            if isinstance(kv_cache, VQCacheOffload):
                for layer_idx in tqdm(range(len(kv_cache.key_cache)), desc="offloading"):
                    kv_cache.key_cache[layer_idx] = kv_cache.key_cache[layer_idx].expand(bsz, -1, -1, -1).contiguous()
                    kv_cache.value_cache[layer_idx] = kv_cache.value_cache[layer_idx].expand(bsz, -1, -1, -1).contiguous()

                    if hasattr(kv_cache, "vq_ids"):
                        kv_cache.vq_ids[layer_idx] = kv_cache.vq_ids[layer_idx].expand(bsz, -1, -1).contiguous()
            else:
                for layer_idx in tqdm(range(len(kv_cache.key_cache)), desc="offloading"):
                    kv_cache.key_cache[layer_idx] = kv_cache.key_cache[layer_idx].expand(bsz, -1, -1, -1).contiguous()
                    kv_cache.value_cache[layer_idx] = kv_cache.value_cache[layer_idx].expand(bsz, -1, -1, -1).contiguous()


            for warmup_idx in tqdm(range(warmup_len), desc="[INFO] warming up"):
                self.hf_model(
                    torch.tensor(input_ids[prefill_len+warmup_idx:prefill_len+warmup_idx+1]).unsqueeze(0).expand(bsz, -1).to(self.device), 
                    past_key_values=kv_cache
                )                  
                kv_cache = output.past_key_values

            torch.cuda.synchronize()
            t_start = time.time()

            for gen_idx in tqdm(range(test_len), "[INFO] benchmarking"):
                output = self.hf_model(
                    torch.tensor(input_ids[prefill_len+warmup_len+gen_idx:prefill_len+warmup_len+gen_idx+1]).unsqueeze(0).expand(bsz, -1).to(self.device), 
                    past_key_values=kv_cache
                )
                kv_cache = output.past_key_values
            
            torch.cuda.synchronize()
            t_end = time.time()

            return (bsz * test_len) / (t_end - t_start)


def llama_attention_forward_wrapper(wrope_local_window,
                                    wrope_relative_pos,
                                    attention_sink,
                                    sampling_ratio,
                                    topk,
                                    decoding_statistics: DecodingStatistics=None
                                    ):
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
        past_key_value: Optional[VQCacheOffload] = None,
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

        def cdist(x, C):
            # x: [bsz, num_key_value_heads, q_len, head_dim]
            # C: [bsz, num_key_value_heads, codebook_size, head_dim]
            x_ = (x**2).sum(dim=-1).unsqueeze(-1) # [bsz, num_key_value_heads, q_len, 1]
            C_ = (C**2).sum(dim=-1).unsqueeze(-2) # [bsz, num_key_value_heads, 1, codebook_size]
            inner_product = torch.matmul(x, C.transpose(-2, -1)) # [bsz, num_key_value_heads, q_len, codebook_size]

            return x_ + C_ - 2 * inner_product

        if hasattr(self, "cholesky_factor"):
            pairwise_distance = cdist(key_states @ self.cholesky_factor, self.codebook)
        else:
            pairwise_distance = cdist(key_states, self.codebook)

        vq_ids = torch.argmin(pairwise_distance, dim=-1)

        if q_len == 1:
            query_states_wrope, _ = apply_rotary_pos_emb(query_states, None, self.cos_wrope, self.sin_wrope)

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
            key_states, value_states, vq_ids = past_key_value.update(key_states, value_states, vq_ids, self.layer_idx, cache_kwargs)

        if q_len == 1:
            if hasattr(self, "inv_cholesky_factor"):
                codebook = self.codebook @ self.inv_cholesky_factor
            else:
                codebook = self.codebook

            attn_weights_codebook = torch.zeros(bsz, self.num_heads, q_len, self.codebook.size(-2), dtype=dtype, device=device)
            
            for key_value_head_idx in range(self.num_key_value_heads):
                head_begin = key_value_head_idx * self.num_key_value_groups
                head_end = (key_value_head_idx + 1) * self.num_key_value_groups
                attn_weights_codebook[:, head_begin:head_end, :, :] = torch.matmul(query_states_wrope[:, head_begin:head_end, :, :], codebook[key_value_head_idx].T)

            attn_weights_approx = attn_weights_codebook.gather(dim=-1, index=vq_ids.repeat_interleave(self.num_key_value_groups, 1).unsqueeze(2))
            
            seq_len = cache_position[-1] + 1

            if topk != 0.0:
                wrope_mask = torch.arange(seq_len - q_len, seq_len, device=device).unsqueeze(1) - torch.arange(seq_len, device=device).unsqueeze(0)
                wrope_mask = wrope_mask < wrope_local_window
                
                attention_sink_mask = torch.zeros((1, seq_len), device=device, dtype=torch.bool)
                attention_sink_mask[:, :attention_sink] = True
                
                attn_weights_approx = torch.where(wrope_mask, float("-inf"), attn_weights_approx)
                attn_weights_approx = torch.where(attention_sink_mask, float("-inf"), attn_weights_approx)
                
                topk_indices = attn_weights_approx.topk(int(topk * seq_len), dim=-1).indices

                sampling_mask = torch.zeros_like(attn_weights_approx, dtype=torch.bool)
                sampling_mask.scatter_(-1, topk_indices, True)
                sampling_mask = torch.where(wrope_mask, True, sampling_mask)
                sampling_mask = torch.where(attention_sink_mask,  True, sampling_mask)

                sampling_mask = sampling_mask.reshape(bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, seq_len).any(dim=-3)

                sampling_mask = sampling_mask.to("cpu")
                query_states_wrope = query_states_wrope.to(torch.float32).to("cpu")
                
                attn_output_sparse_kernel = selective_attention.launch(
                    query_states_wrope,
                    key_states,
                    value_states,
                    sampling_mask,
                )
                
                # attn_output = torch.zeros((bsz, self.num_heads, q_len, self.head_dim), device="cpu", dtype=dtype)
                # for batch_idx in range(bsz):
                #     for key_value_head_idx in range(self.num_key_value_heads):
                #         recall_indices = sampling_mask[batch_idx, key_value_head_idx, 0].nonzero().squeeze(-1)
                #         key_states_recalled, value_states_recalled = key_states[batch_idx, key_value_head_idx, recall_indices], value_states[batch_idx, key_value_head_idx, recall_indices]

                #         attention_head_begin = key_value_head_idx * self.num_key_value_groups
                #         attention_head_end = (key_value_head_idx + 1) * self.num_key_value_groups
                        
                #         attn_weights = torch.matmul(query_states_wrope[batch_idx, attention_head_begin:attention_head_end], key_states_recalled.transpose(0, 1)) / math.sqrt(self.head_dim)
                #         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
                #         attn_output[batch_idx, attention_head_begin:attention_head_end] = torch.matmul(attn_weights, value_states_recalled)

                # assert torch.allclose(attn_output, attn_output_sparse_kernel, atol=1e-4)
                
                attn_output = attn_output_sparse_kernel.to(device).to(dtype)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)

            else:
                raise Exception("[ERROR] invalid sampling parameters")
        
        else:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            
            assert attention_mask is not None
            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                is_causal=True
            )

            past_key_value.offload(self.layer_idx)

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


class VQCacheOffload(Cache):
    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = None,
        max_cache_len: int = None,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        max_batch_size: Optional[int] = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__()
        if max_batch_size is not None:
            logger.warning_once(
                f"The 'max_batch_size' argument of {self.__class__.__name__} is deprecated and will be removed in "
                "v4.46. Use the more precisely named 'batch_size' argument instead."
            )

        self.batch_size = batch_size or max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len

        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.vq_ids: List[torch.Tensor] = []

        self.key_cache_cpu: List[torch.Tensor] = []
        self.value_cache_cpu: List[torch.Tensor] = []

        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        cache_shape = (self.batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        for _ in range(config.num_hidden_layers):
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            new_layer_vq_ids = torch.zeros(cache_shape[ : -1], dtype=torch.int64, device=device)

            new_layer_key_cache_cpu = torch.zeros(cache_shape, dtype=torch.float32, device="cpu").pin_memory()
            new_layer_value_cache_cpu = torch.zeros(cache_shape, dtype=torch.float32, device="cpu").pin_memory()

            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)
            self.vq_ids.append(new_layer_vq_ids)

            self.key_cache_cpu.append(new_layer_key_cache_cpu)
            self.value_cache_cpu.append(new_layer_value_cache_cpu)
        
        self.seq_len = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        vq_ids,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")

        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        vq_ids_out = self.vq_ids[layer_idx]

        if cache_position is None:
            raise Exception("[ERROR] cache_position is None")

        device = self.key_cache[layer_idx].device
        dtype = self.key_cache[layer_idx].dtype

        k_out[:, :, cache_position] = key_states.to(dtype).to(device)
        v_out[:, :, cache_position] = value_states.to(dtype).to(device)

        vq_ids_out[:, :, cache_position] = vq_ids

        self.seq_len = cache_position[-1] + 1
        return k_out[:, :, :self.seq_len], v_out[:, :, :self.seq_len], vq_ids_out[:, :, :self.seq_len]

    def offload(
            self,
            layer_idx
    ):
        self.key_cache_cpu[layer_idx].copy_(self.key_cache[layer_idx].to(torch.float32), non_blocking=True)
        self.value_cache_cpu[layer_idx].copy_(self.value_cache[layer_idx].to(torch.float32), non_blocking=True)

    def switch_between_cpu_and_gpu(self):
        self.key_cache = self.key_cache_cpu
        self.value_cache = self.value_cache_cpu

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        
        # return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()
        return self.seq_len

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()