python -m vllm.entrypoints.openai.api_server \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 500 \
  --max-num-seqs 8 \
  --port 8000

