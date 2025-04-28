#!/bin/bash
set -e

repo="/blue/buyuheng/li_an.ucsb/projects/rl-watermark"

embed_map_model="/blue/buyuheng/li_an.ucsb/projects/rl-watermark/ckpts/batch16-nmini2-G8-sanity-detect1/embed_map_model_best"
watermark_model="meta-llama/Llama-3.1-8B-Instruct"
alpha=1.0
delta_0=0.1
delta=0.13

data_path="Shiyu-Lab/C4-contrastive-watermark"
train_batch_size=16

watermark_output_file="outputs/$(basename "$(dirname "$embed_map_model")")/$(basename "$watermark_model")-alpha${alpha}-delta${delta_0}|${delta}.csv"

# ===== watermarking =====
python $repo/watermarking.py \
    --embed_map_model=$embed_map_model \
    --watermark_model=$watermark_model \
    --output_file=${watermark_output_file} \
    --alpha=${alpha} --delta_0=$delta_0 --delta=$delta \
    --data_path=${data_path} \
    --data_size=${train_batch_size} \
