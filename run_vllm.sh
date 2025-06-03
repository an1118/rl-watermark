#!/bin/bash
#SBATCH --job-name=watermark
#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
##SBATCH --reservation=buyuheng 
#SBATCH --gpus=a100:4
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00

# module load cuda
set -e

# vllm_log_file="vllm-${SLURM_JOB_ID}.log"
vllm_log_file=outputs/vllm.log

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "Qwen/Qwen3-14B" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 2000 \
  --max-num-seqs 8 \
  --port 8000 > "$vllm_log_file" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "Terminating vLLM server (PID=$VLLM_PID)..."
  kill $VLLM_PID 2>/dev/null || true
  wait $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

sleep 60

READY=0
for i in {1..10}; do
  if nc -z localhost 8000; then
    echo "vLLM server is ready."
    READY=1
    break
  fi
  sleep 10
done

if [ "$READY" -ne 1 ]; then
  echo "vLLM server failed to start in time." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=1,2,3 python grpo.py
