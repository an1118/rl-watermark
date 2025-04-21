# vllm_runner.py
import os
from vllm import LLM

os.environ["VLLM_USE_V1"] = "0"

model = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    max_model_len=500,
)

