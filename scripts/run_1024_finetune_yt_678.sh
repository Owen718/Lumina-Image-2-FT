
export NCCL_SOCKET_IFNAME=eth100
export NCCL_SOCKET_IFNAME=eth100,eth101
export NCCL_SOCKET_IFNAME=rdma0,rdma1,rdma2,rdma3
export NCCL_IB_DISABLE=1


#!/usr/bin/env sh
cd /mnt/weka/yt_workspace/Lumina-Image-2.0
# --- Configuration ---
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1
train_data_path='./configs/data.yaml'


model=NextDiT_2B_GQA_patch2_Adaln_Refiner
check_path=/mnt/weka/yt_workspace/Lumina-Image-2.0/model_weights
# batch_size=16  # This is effectively the global batch size
snr_type=lognorm
lr=4e-5
precision=bf16
size=1024

exp_name=${model}_lr${lr}_${precision}_gemma3-4b-it-2
mkdir -p results/"$exp_name"

# --- Distributed Training Configuration (torchrun) ---
NNODES=1
NPROC_PER_NODE=4  # Use 2 GPUs (0 and 1)
MASTER_PORT=18182 # Keep the same port if no conflict
NODE_RANK=0

# --- torchrun command ---
torchrun \
    --nnodes ${NNODES} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --master_port ${MASTER_PORT} \
    finetune.py \
    --global_bsz_${size} 64 \
    --micro_bsz_${size} 8 \
    --model ${model} \
    --lr ${lr} --grad_clip 2.0 \
    --data_path ${train_data_path} \
    --results_dir results/"$exp_name" \
    --data_parallel sdp \
    --max_steps 3000000 \
    --ckpt_every 500 --log_every 10 \
    --precision ${precision} --grad_precision fp32 --qk_norm \
    --global_seed 20241207 \
    --num_workers 12 \
    --snr_type ${snr_type} \
    --checkpointing \
    --init_from ${check_path} \
    2>&1 | tee -a results/"$exp_name"/output.log
