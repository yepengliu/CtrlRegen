#!/bin/bash
# Stage A -- train the Semantic Control adapter.
# Adjust --num_processes to your GPU count. The released weights used 8 x A100 (batch 8 each).
set -e

accelerate launch --num_processes 8 --multi_gpu --mixed_precision "fp16" \
  train_semantic.py \
  --pretrained_model_name_or_path="SG161222/Realistic_Vision_V4.0_noVAE" \
  --image_encoder_path="facebook/dinov2-giant" \
  --data_json_file="data.json" \
  --data_root_path="" \
  --output_dir="out_semantic" \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-4 \
  --weight_decay=0.01 \
  --num_train_epochs=100 \
  --save_steps=5000 \
  --mixed_precision="fp16" \
  --report_to="wandb"

# To resume from a converted checkpoint (e.g. after a job time limit), add:
#   --pretrained_ip_adapter_path="out_semantic/semantic_control_ckp_<step>.bin"
