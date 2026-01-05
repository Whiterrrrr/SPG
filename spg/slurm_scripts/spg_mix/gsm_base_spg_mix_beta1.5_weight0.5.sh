#!/bin/bash
export WANDB_API_KEY=8d739e5eaa28091db300de37eb709020ff7cf27c
export LOGDIR=../../logs
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=True
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SAVE_DIR=/data/discrete-diffusionRL-LLM

export LOGDIR=../../logs
mkdir -p $LOGDIR
echo $LOGDIR

source activate spg
# Generate a random port number between 10000 and 65535
RANDOM_PORT=$((RANDOM % 55536 + 10000))
echo "Using random main_process_port: $RANDOM_PORT"

DATASET="gsm8k"
RUN_NAME=${DATASET}_base_spg_mix_beta1.5_weight0.5
MODEL_PATH=${SAVE_DIR}/hf_models/LLaDA-8B-Instruct
NUM_ITER=4

sudo -E /home/zhengkx/.conda/envs/spg/bin/python -m accelerate.commands.launch \
    --config_file ../accelerate_genai_a100.yaml \
    --main_process_port $RANDOM_PORT ../../diffu_grpo_train.py \
    --config ../train.yaml \
    --model_path $MODEL_PATH \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir ${SAVE_DIR}/spg/$RUN_NAME \
    --trainer spg \
    --forward_type block_random \
    --num_t 2 \
    --min_t 0 \
    --max_t 1 \
    --num_generations 6 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --beta 0.0 \
    --logp_estimation mix \
    --mix_weight 0.5 \
    --eubo_beta 1.5 \
    