#!/bin/bash

DEFAULT_RUN_NAME="AgroNVILA_SFT"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=32
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Use $HOME instead of ~ : a tilde inside a default expansion is not tilde-expanded.
STAGE_PATH=${1:-"$HOME/model/NVILA-Lite-8B"}
DATA_MIXTURE=${2:-"AgroOmni"}
OUTPUT_DIR=${3:-"runs/train/${DEFAULT_RUN_NAME}"}

source scripts/setups/train.sh

mkdir -p $OUTPUT_DIR/logs
LOG_FILE="$OUTPUT_DIR/logs/training.log"

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $STAGE_PATH \
        --data_mixture $DATA_MIXTURE \
        --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample_3x3_fix \
        --tune_vision_tower False \
        --tune_mm_projector True \
        --tune_language_model False \
        --lora_enable True \
        --modules_to_save "mm_projector" \
        --lora_llm True \
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --enable_view False \
        --image_aspect_ratio dynamic \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 900 \
        --save_total_limit 10 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 5120 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --vflan_no_system_prompt True \
        --report_to wandb 2>&1 | tee "$LOG_FILE"
