CUDA_VISIBLE_DEVICES=0 vllm serve "/blue/buyuheng/li_an.ucsb/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/8268fe3026cb304910457689366670e803a6fd56" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 2000 \
  --max-num-seqs 128 \
  --port 8000


# vllm serve "meta-llama/Llama-3.1-70B" \
#   --tensor-parallel-size 1 \
#   --dtype bfloat16 \
#   --max-model-len 600 \
#   --max-num-seqs 16 \
#   --port 6666
