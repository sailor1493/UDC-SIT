#!/bin/bash
#SBATCH --job-name ;5
#SBATCH --partition chanwoo
#SBATCH --output logs/%j.log
#SBATCH --error logs/%j.err
#SBATCH --exclusive

# make unique directory for each pid
# assume that slurm jobid is unavailable
random_id=$(date +%s)
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache-${random_id}"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# make random port using pid
port=$((10000 + $RANDOM % 20000))
echo "Port: $port"
export MASTER_PORT=$port



phase=$1
if [[ -z $phase ]]; then
    phase=1
fi

loss_lambda=0.5
if [[ -z $loss_lambda ]]; then
    loss_lambda=0.5
fi

num_channels=4
num_res=8
loss_version=v2
mapping=norm

base_dir=/shared/s1/lab08/udc/UDC-SIT
train_input="$base_dir/training/input"
train_gt="$base_dir/training/GT"
val_input="$base_dir/validation/input"
val_gt="$base_dir/validation/GT"

level=6
wandb_name="mi100-intensified-level$level-develop"
step_1="${wandb_name}-1"
step_2="${wandb_name}-2"
step_3="${wandb_name}-3"


# 1000
if [[ $phase -le 1 ]]; then
    echo "$0 loss lambda $loss_lambda started Phase 1"
    python train_intensified.py \
        --train-input $train_input \
        --train-gt $train_gt \
        --val-input $val_input \
        --val-gt $val_gt \
        --patch-size 256 \
        --batch-size 4 \
        --loss-lambda $loss_lambda \
        --experiment-name $step_1 \
        --num-workers 8 \
        --num-epochs 1 \
        --validation-interval 1 \
        --save-interval 1 \
        --lr 0.0002 \
        --num-res $num_res \
        --loss-version $loss_version \
        --mapping $mapping \
        --wandb-name $wandb_name \
        --level-ablation $level \
        --channels $num_channels \
        --save-img-count 1 \
        --aggressive-checkpointing

    if [[ $? -ne 0 ]]; then
        echo "############################################################"
        echo "$0 loss lambda $loss_lambda Phase 1 failed"
        echo "############################################################"
        exit 1
    fi
fi

if [[ $phase -le 2 ]]; then

    echo "$0 loss lambda $loss_lambda Phase 2 started"

    # 300
    python train_intensified.py \
        --train-input $train_input \
        --train-gt $train_gt \
        --val-input $val_input \
        --val-gt $val_gt \
        --patch-size 512 \
        --batch-size 1 \
        --loss-lambda $loss_lambda \
        --experiment-name $step_2 \
        --num-workers 8 \
        --num-epochs 1 \
        --validation-interval 1 \
        --save-interval 1 \
        --lr 0.00001 \
        --pretrained-model experiments/$step_1/model_best.pth  \
        --num-res $num_res \
        --loss-version $loss_version \
        --mapping $mapping \
        --wandb-name $wandb_name \
        --level-ablation $level \
        --channels $num_channels \
        --save-img-count 1 \
        --aggressive-checkpointing

    if [[ $? -ne 0 ]]; then
        echo "############################################################"
        echo "$0 loss lambda $loss_lambda Phase 2 failed"
        echo "############################################################"
        exit 1
    fi
fi

if [[ $phase -le 3 ]]; then
    echo "$0 loss lambda $loss_lambda Phase 3 started"

    # 150
    python train_intensified.py \
        --train-input $train_input \
        --train-gt $train_gt \
        --val-input $val_input \
        --val-gt $val_gt \
        --patch-size 800 \
        --batch-size 1 \
        --loss-lambda $loss_lambda \
        --experiment-name $step_3 \
        --num-workers 8 \
        --num-epochs 1 \
        --validation-interval 1 \
        --save-interval 1 \
        --lr 0.000008 \
        --aggressive-checkpointing \
        --num-res $num_res \
        --loss-version $loss_version \
        --mapping $mapping \
        --wandb-name $wandb_name-fixed \
        --pretrained-model experiments/$step_2/model_best.pth \
        --level-ablation $level \
        --channels $num_channels \
        --save-img-count 1

    if [[ $? -ne 0 ]]; then
        echo "############################################################"
        echo "$0 loss lambda $loss_lambda Phase 3 failed"
        echo "############################################################"
        exit 1
    fi

    echo "$0 loss lambda $loss_lambda finished"
fi

