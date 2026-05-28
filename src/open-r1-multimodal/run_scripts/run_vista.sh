cd src/open-r1-multimodal

export TASK_ALGO=vista
export TASK_TYPE=grounding
export TASK_MODEL=qwen3

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

RUN_NAME="exp_id_${TASK_ALGO}_${TASK_TYPE}_${TASK_MODEL}"

export DATA_PATH=data_config/dataset.yaml
export PYTHONPATH=src

export CKPT_PATH=/root/Qwen/Qwen3-VL-8B-Instruct/

# 定义目标路径
export OUTPUT_ROOT=/root/${TASK_ALGO}_qwen3vl_${TASK_TYPE}
export SAVE_PATH=${OUTPUT_ROOT}/${RUN_NAME}_${TIMESTAMP}/
export LOG_DIR=${OUTPUT_ROOT}/${RUN_NAME}_${TIMESTAMP}/

echo $SAVE_PATH

mkdir -p "${LOG_DIR}"

export LOG_PATH="${LOG_DIR}/log_${TIMESTAMP}_out.txt"
export WANDB_DIR="${LOG_DIR}"

WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)

echo "N_NODE: $N_NODE"
echo "N_GPU_PER_NODE: $N_GPU_PER_NODE"
echo "LOG_DIR: $LOG_DIR"
echo "TASK_MEMO: $TASK_MEMO"
echo "DATA_PATH: $DATA_PATH"
echo "SAVE_PATH: $SAVE_PATH"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

{
    echo "N_NODE: $N_NODE"
    echo "N_GPU_PER_NODE: $N_GPU_PER_NODE"
    echo "LOG_DIR: $LOG_DIR"
    echo "TASK_MEMO: $TASK_MEMO"
    echo "DATA_PATH: $DATA_PATH"
    echo "SAVE_PATH: $SAVE_PATH"
    echo "WORLD_SIZE: $WORLD_SIZE"
    echo "RANK: $RANK"
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"
} > "$LOG_PATH"


DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_COUNT \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${RANK} \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


torchrun $DISTRIBUTED_ARGS src/open_r1/vista.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --image_root /path/to/image/ \
    --max_prompt_length 4096 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --freeze_vision_modules true \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 100 \
    --run_name $RUN_NAME \
    --save_steps 300 \
    --max_pixels 12845056 \
    --save_only_model true \
    --save_total_limit 5 \
    --beta 0.04  \
    --learning_rate 1e-6 $@ 2>&1 | tee "${LOG_DIR}/log_${TIMESTAMP}.log"

# --beta 0  \
