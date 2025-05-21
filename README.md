# A2ATS: Retrieval-Based KV Cache Reduction via Windowed Rotary Position Embedding and Query-Aware Vector Quantization

📃Junhui He, Junna Xing, Nan Wang, Rui Xu, Shangyu Wu, Peng Zhou, Qiang Liu, Chun Jason Xue, Qingan Li:
A2ATS: Retrieval-Based KV Cache Reduction via Windowed Rotary Position Embedding and Query-Aware Vector Quantization. **ACL 2025 Findings** 🔗[Link](https://arxiv.org/abs/2502.12665)

## Requirements

- NVIDIA GPU with at least 40GB VRAM for codebook construction

## Install Dependencies

```bash
# (Recommended) Create conda environment
conda create --name vqkey "python<3.13"
conda activate vqkey

# Install dependencies
pip install -r requirements.txt
pip install -e ./vqkey/kmeans-gpu
```

## Prepare the A$^2$ATS Models

```bash
# Download the fineweb-edu calibration dataset
huggingface-cli download --repo-type dataset HuggingFaceFW/fineweb-edu --include sample/10BT/000_00000.parquet --local-dir huggingface-datasets/fineweb-edu

# Download the Llama-3.1-8B-Instruct model
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir huggingface-models/Llama-3.1-8B-Instruct --exclude original/*
```

```bash
# Construct codebooks
cd ./vqkey/notebooks
python ./construct_codebooks.py
cd ../..
```

## Run the A$^2$ATS Models

Import and initialize tokenizer and our custom model:

```python
from models.llama_masked import LLM
from transformers import AutoTokenizer

model_dir = "../huggingface-models/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = LLM(model_dir, "./codebooks", "llama-3-8b-codebooks.pt", "llama-3-8b-cholesky_factors.pt", topk=0.03)
```

Text generation using our customized model:

```python
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

output_ids = model.generate(inputs)

print(tokenizer.decode(output_ids))
```

Our custom sparse attention kernel `selective_attention` used by `models.llama` has shown good performance on the Intel Xeon Platinum 8469C CPU, but performance on other CPUs may still need optimization and could encounter compatibility issues. We are actively optimizing our custom kernels to achieve the best efficiency on a broad range of CPUs. We will release these kernels once they become available. In the meantime, please use `models.llama_masked` as a mathematically equivalent fallback for accuracy evaluation.