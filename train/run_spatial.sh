#!/bin/bash
# Stage B -- train the Spatial Control network.
# Requires a Stage A checkpoint (converted with convert_checkpoint.py --stage semantic).
set -e

accelerate launch --num_processes 8 --multi_gpu --mixed_precision "fp16" \
  train_spatial.py \
  --pretrained_model_name_or_path="SG161222/Realistic_Vision_V4.0_noVAE" \
  --controlnet_model_name_or_path="lllyasviel/control_v11p_sd15_canny" \
  --image_encoder_path="facebook/dinov2-giant" \
  --ip_checkpoint_path="semanticnet_ckp/models/semantic_control_ckp_435000.bin" \
  --dataset_name="AmritaBha/mscoco-controlnet-canny" \
  --image_column="image" \
  --conditioning_image_column="conditioning_image" \
  --caption_column="text" \
  --output_dir="out_spatial" \
  --resolution=512 \
  --train_batch_size=4 \
  --num_train_epochs=4 \
  --learning_rate=1e-5 \
  --proportion_empty_prompts=1.0 \
  --checkpointing_steps=1000 \
  --validation_steps=10000000 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --tracker_project_name="ctrlregen-spatial"
