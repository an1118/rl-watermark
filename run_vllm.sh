#!/bin/bash
#SBATCH --job-name=watermark
#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err
#SBATCH --nodes=1
#SBATCH --partition=hpg-b200
##SBATCH --reservation=buyuheng 
#SBATCH --gpus=4
#SBATCH --mem=64gb
#SBATCH --time=3-00:00:00
##SBATCH --exclude=c0903a-s25

# module load cuda
set -e

echo "=== GPU Status at Job Start ==="
nvidia-smi
echo "==============================="

vllm_log_file="outputs/${SLURM_JOB_ID}.vllm"

VLLM_PORT=$((SLURM_JOB_ID % 65535))
CUDA_VISIBLE_DEVICES=0 vllm serve "/blue/buyuheng/li_an.ucsb/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/8268fe3026cb304910457689366670e803a6fd56" \
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
for i in {1..20}; do
  if nc -z localhost ${VLLM_PORT}; then
    echo "vLLM server is ready."
    READY=1
    break
  fi
  sleep 30
done

if [ "$READY" -ne 1 ]; then
  echo "vLLM server failed to start in time." >&2
  exit 1
fi

repo="/blue/buyuheng/li_an.ucsb/projects"
github_repo="git@github.com:an1118/rl-watermark.git"
branch="sanity-detect_attack-v2" # sanity-detect_attack-v2 embed_vocab_size

watermark_model_name="meta-llama/Llama-3.1-8B-Instruct"  # Qwen/Qwen3-8B meta-llama/Llama-3.1-8B-Instruct
is_sanity_check=false 
seed=666
log_grad_norm=true

max_step=1000
batch_size=16
num_minibatches=2
G=8  # 8
clip_coef=0.2
beta=0.04

learning_rate=1e-5
lr_scheduler_type=constant
warmup_steps=0

binary=false  # if true, how to add second gradient
use_soft_split=false
use_median_split=false
add_reward_gradient=false
add_gr_loss=false
add_similarity_loss=false
curriculum="none"
detect_steps=6
spoof_steps=6
detect_score_coefs_ori=1
ori_score_strategy="smooth_gap"  # [raw, abs, dynamic, gap, smooth_gap]
target_ori_score=0.5
ori_growth_rate=50
ori_growth_rate2=250
detect_score_coefs_wm=1
wm_score_strategy="raw"
wm_growth_rate=0.1
detect_score_coefs_para=1
para_score_strategy="raw"
para_growth_rate=0.2
detect_score_coefs_senti=1
# detect_score_coefs_latter=1
detect_score_coefs_hate=1
ppl_coef=0

do_eval=true
eval_steps=20  # 20
eval_batch_size=100  # 100


run_id="batch$batch_size-nmini$num_minibatches-G$G-clip$clip_coef-beta$beta-lr_${learning_rate}_${lr_scheduler_type}_${warmup_steps}"
model_name=$(echo "$watermark_model_name" | awk -F'/' '{print $2}')
if [ -z "$model_name" ]; then
  echo "Failed to extract model name from watermark_model_name: $watermark_model_name" >&2
  exit 1
fi
run_id="${model_name}-${run_id}"
if [ "${curriculum,,}" = "none" ]; then
  run_id="${run_id}-ori${detect_score_coefs_ori}(${ori_score_strategy})wm${detect_score_coefs_wm}(${wm_score_strategy})para${detect_score_coefs_para}(${para_score_strategy})senti${detect_score_coefs_senti}hate${detect_score_coefs_hate}"
else
  run_id="${run_id}-ct_${curriculum}_d${detect_steps}s${spoof_steps}_ori(${ori_score_strategy})wm(${wm_score_strategy})para(${para_score_strategy})"
fi
if (( $(echo "$ppl_coef > 0.0" | bc -l) )); then
  run_id="${run_id}-ppl${ppl_coef}"
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
if [ "$add_similarity_loss" = true ]; then
    run_id="${run_id}-sim_loss"
fi
if [ "$ori_score_strategy" = "abs" ]; then
  run_id=$(echo "$run_id" | sed -E "s/(ori[0-9]*)\(${ori_score_strategy}\)/\1(${ori_score_strategy}-${target_ori_score})/")
fi
if [ "$ori_score_strategy" = "dynamic" ] || [ "$ori_score_strategy" = "gap" ]; then
  run_id=$(echo "$run_id" | sed -E "s/(ori[0-9]*)\(${ori_score_strategy}\)/\1(${ori_score_strategy}-${ori_growth_rate})/")
fi
if [ "$ori_score_strategy" = "smooth_gap" ]; then
  run_id=$(echo "$run_id" | sed -E "s/(ori[0-9]*)\(${ori_score_strategy}\)/\1(${ori_score_strategy}-${ori_growth_rate}-${ori_growth_rate2})/")
fi
if [ "$wm_score_strategy" = "dynamic" ]; then
  run_id=$(echo "$run_id" | sed -E "s/(wm[0-9]*)\(${wm_score_strategy}\)/\1(${wm_score_strategy}-${wm_growth_rate})/")
fi
if [ "$para_score_strategy" = "dynamic" ]; then
  run_id=$(echo "$run_id" | sed -E "s/(para[0-9]*)\(${para_score_strategy}\)/\1(${para_score_strategy}-${para_growth_rate})/")
fi
version=$(git ls-remote --refs $github_repo $branch | awk '{print substr($1,1,7)}')
run_id="${run_id}-${version}-seed${seed}"
echo "Run ID: $run_id"
clone_dir="$repo/tmp/$run_id"
rm -rf $clone_dir

git clone --branch $branch --single-branch $github_repo $clone_dir
cd $clone_dir
cp /blue/buyuheng/li_an.ucsb/projects/rl-watermark/api.py $clone_dir/api.py

CUDA_VISIBLE_DEVICES=1,2,3 python grpo.py \
  --seed $seed \
  --watermark_model_name $watermark_model_name \
  --max_step $max_step \
  --batch_size $batch_size \
  --num_minibatches $num_minibatches \
  --G $G \
  --clip_coef $clip_coef \
  --beta $beta \
  --learning_rate $learning_rate \
  --lr_scheduler_type $lr_scheduler_type \
  --warmup_steps $warmup_steps \
  --checkpoint_dir $repo/rl-watermark/ckpts/$run_id \
  --run_name $run_id \
  --curriculum $curriculum \
  --detect_steps $detect_steps \
  --spoof_steps $spoof_steps \
  --detect_score_coefs_ori $detect_score_coefs_ori \
  --ori_score_strategy $ori_score_strategy \
  --target_ori_score $target_ori_score \
  --ori_growth_rate $ori_growth_rate \
  --ori_growth_rate2 $ori_growth_rate2 \
  --detect_score_coefs_wm $detect_score_coefs_wm \
  --wm_score_strategy $wm_score_strategy \
  --wm_growth_rate $wm_growth_rate \
  --detect_score_coefs_para $detect_score_coefs_para \
  --para_score_strategy $para_score_strategy \
  --para_growth_rate $para_growth_rate \
  --detect_score_coefs_senti $detect_score_coefs_senti \
  --detect_score_coefs_hate $detect_score_coefs_hate \
  --ppl_coef $ppl_coef \
  --eval_steps $eval_steps \
  --eval_batch_size $eval_batch_size \
  --attack_model_name "Qwen/Qwen3-14B" \
  --attack_model_url "http://localhost:${VLLM_PORT}/v1" \
  $( [ "$is_sanity_check" = true ] && echo "--is_sanity_check" ) \
  $( [ "$do_eval" = true ] && echo "--do_eval" ) \
  $( [ "$binary" = true ] && echo "--binary" ) \
  $( [ "$use_soft_split" = true ] && echo "--use_soft_split" ) \
  $( [ "$use_median_split" = true ] && echo "--use_median_split" ) \
  $( [ "$add_reward_gradient" = true ] && echo "--add_reward_gradient" ) \
  $( [ "$add_gr_loss" = true ] && echo "--add_gr_loss" )
  $( [ "$add_similarity_loss" = true ] && echo "--add_similarity_loss" )

# CUDA_VISIBLE_DEVICES=1,2,3 python grpo.py