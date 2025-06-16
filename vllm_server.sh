CUDA_VISIBLE_DEVICES=0 vllm serve "Qwen/Qwen3-14B" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 2000 \
  --max-num-seqs 128 \
  --port 8000