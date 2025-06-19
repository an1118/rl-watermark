#!/bin/bash
#SBATCH --job-name=watermark
#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err
#SBATCH --nodes=1
#SBATCH --partition=hpg-b200
##SBATCH --reservation=buyuheng 
#SBATCH --gpus=4
#SBATCH --mem=128gb
#SBATCH --time=5-00:00:00
##SBATCH --exclude=c0903a-s25

# module load cuda
set -e

vllm_log_file="outputs/${SLURM_JOB_ID}.vllm"

VLLM_PORT=$((SLURM_JOB_ID % 65535))
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "Qwen/Qwen3-14B" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 2000 \
  --max-num-seqs 128 \
  --port ${VLLM_PORT}  > "$vllm_log_file" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "Terminating vLLM server (PID=$VLLM_PID)..."
  kill $VLLM_PID 2>/dev/null || true
  wait $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

sleep 180

READY=0
for i in {1..10}; do
  if nc -z localhost ${VLLM_PORT}; then
    echo "vLLM server is ready."
    READY=1
    break
  fi
  sleep 20
done

if [ "$READY" -ne 1 ]; then
  echo "vLLM server failed to start in time." >&2
  exit 1
fi

repo="/blue/buyuheng/li_an.ucsb/projects"
github_repo="https://github.com/an1118/rl-watermark.git"
branch="sanity-detect_attack-v2"

is_sanity_check=false 
seed=666

max_step=500
batch_size=16  # 64
num_minibatches=2
G=8  # 8
clip_coef=0.2
beta=0.04

binary=false  # if true, how to add second gradient
use_soft_split=false
use_median_split=false
add_reward_gradient=true
add_gr_loss=false
curriculum="v1"
curriculum_steps=6
detect_score_coefs_ori=1
ori_score_strategy="gap"  # [raw, abs, dynamic, gap]
target_ori_score=0.5
growth_rate=0.8
sharpness=5
detect_score_coefs_wm=1
detect_score_coefs_para=1
detect_score_coefs_senti=1
# detect_score_coefs_latter=1
detect_score_coefs_hate=1

do_eval=true
eval_steps=20  # 20
eval_batch_size=100  # 100


run_id="batch$batch_size-nmini$num_minibatches-G$G-clip$clip_coef-beta$beta"
if [ -n "$curriculum" ] && [ "${curriculum,,}" = "none" ]; then
  run_id="${run_id}-ori${detect_score_coefs_ori}(${ori_score_strategy})wm${detect_score_coefs_wm}para${detect_score_coefs_para}senti${detect_score_coefs_senti}hate${detect_score_coefs_hate}"
else
  run_id="${run_id}-ct_${curriculum}_step${curriculum_steps}_ori(${ori_score_strategy})"
fi
if [ "$is_sanity_check" = true ]; then
    run_id="sanity_check-${run_id}"
fi
if [ "$binary" = true ]; then
    run_id="${run_id}-binary"
fi
if [ "$use_soft_split" = true ]; then
    run_id="${run_id}-soft"
fi
if [ "$use_median_split" = true ]; then
    run_id="${run_id}-median"
fi
if [ "$add_reward_gradient" = true ]; then
    run_id="${run_id}-reward_gradient"
fi
if [ "$add_gr_loss" = true ]; then
    run_id="${run_id}-gr_loss"
fi
if [ "$ori_score_strategy" = "abs" ]; then
  run_id=$(echo "$run_id" | sed "s/(${ori_score_strategy})/(${ori_score_strategy}-${target_ori_score})/")
fi
if [ "$ori_score_strategy" = "dynamic" ]; then
  run_id=$(echo "$run_id" | sed "s/(${ori_score_strategy})/(${ori_score_strategy}-${growth_rate})/")
fi
if [ "$ori_score_strategy" = "gap" ]; then
  run_id=$(echo "$run_id" | sed "s/(${ori_score_strategy})/(${ori_score_strategy}-${sharpness})/")
fi
version=$(git ls-remote --refs $github_repo $branch | awk '{print substr($1,1,7)}')
run_id="${run_id}-${version}-seed${seed}"
echo "Run ID: $run_id"
clone_dir="$repo/scratch/$run_id"
rm -rf $clone_dir

git clone --branch $branch --single-branch $github_repo $clone_dir
cd $clone_dir
cp /blue/buyuheng/li_an.ucsb/projects/rl-watermark/api.py $clone_dir/api.py

CUDA_VISIBLE_DEVICES=1,2,3 python grpo.py \
  --max_step $max_step \
  --batch_size $batch_size \
  --num_minibatches $num_minibatches \
  --G $G \
  --clip_coef $clip_coef \
  --beta $beta \
  --checkpoint_dir $repo/rl-watermark/ckpts/$run_id \
  --run_name $run_id \
  --curriculum $curriculum \
  --curriculum_steps $curriculum_steps \
  --detect_score_coefs_ori $detect_score_coefs_ori \
  --ori_score_strategy $ori_score_strategy \
  --target_ori_score $target_ori_score \
  --growth_rate $growth_rate \
  --sharpness $sharpness \
  --detect_score_coefs_wm $detect_score_coefs_wm \
  --detect_score_coefs_para $detect_score_coefs_para \
  --detect_score_coefs_senti $detect_score_coefs_senti \
  --detect_score_coefs_hate $detect_score_coefs_hate \
  --eval_steps $eval_steps \
  --eval_batch_size $eval_batch_size \
  --attack_model_name "Qwen/Qwen3-14B" \
  --attack_model_url "http://localhost:${VLLM_PORT}/v1" \
  $( [ "$is_sanity_check" = true ] && echo "--is_sanity_check" ) \
  $( [ "$do_eval" = true ] && echo "--do_eval" ) \
  $( [ "$binary" = true ] && echo "--binary" ) \
  $( [ "$use_soft_split" = true ] && echo "--use_soft_split" ) \
  $( [ "$add_reward_gradient" = true ] && echo "--add_reward_gradient" ) \
  $( [ "$add_gr_loss" = true ] && echo "--add_gr_loss" )

# CUDA_VISIBLE_DEVICES=1,2,3 python grpo.py