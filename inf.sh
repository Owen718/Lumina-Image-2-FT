#!/bin/bash

# 模型检查点路径（必需）
# CHECKPOINT_PATH="/mnt/weka/yt_workspace/Lumina-Image-2.0/results/NextDiT_2B_GQA_patch2_Adaln_Refiner_lr2e-5_bf16/checkpoints/0004000"
CHECKPOINT_PATH="/mnt/weka/yt_workspace/Lumina-Image-2.0/results/NextDiT_2B_GQA_patch2_Adaln_Refiner_lr4e-5_bf16_gemma3-4b-it-2/checkpoints/0001500"
# 基本配置
SOLVER="euler"           # 求解器类型: euler, dpm 等
NUM_STEPS=40            # 采样步数
CFG_SCALE=4.0            # CFG 缩放系数
PRECISION="bf16"         # 精度: bf16 或 fp32
VAE="flux"               # VAE 类型: flux, ema, mse, sdxl

# 提示词和输出配置
CAPTION_PATH="prompt.txt"  # 提示词文件路径
IMAGE_SAVE_PATH="./results/NextDiT_2B_GQA_patch2_Adaln_Refiner_lr2e-5_bf16_gemma3-4b-it-2"   # 图像保存路径
RESOLUTION="1:1024x1024"    # 分辨率 (格式: res_cat:widthxheight)
SYSTEM_TYPE="real"          # 系统提示类型
export CUDA_VISIBLE_DEVICES=5
# 执行推理
python sample.py \
  --ckpt $CHECKPOINT_PATH \
  --solver $SOLVER \
  --num_sampling_steps $NUM_STEPS \
  --cfg_scale $CFG_SCALE \
  --precision $PRECISION \
  --vae $VAE \
  --caption_path $CAPTION_PATH \
  --image_save_path $IMAGE_SAVE_PATH \
  --resolution $RESOLUTION \
  --system_type $SYSTEM_TYPE \
  --text_encoder gemma3 \
  --time_shifting_factor 1.0 \
  --t_shift 6