#!/bin/bash


pstring=0,1,2

name="Feng" # dataset name
channels=3 # feng dataset has 3 channels
norm="tonemap" # feng dataset use tone mapping for normalization

test_gt="/data/s1/chanwoo/udc_data/test/GT"
test_input="/home/n2/chanwoo/UDC/cvpr_rebuttal/DISCNet/datasets/synthetic_data/input/ZTE_new_5/test"
save_dir="results/${name}"

# Ours: nres 6 version
model_path="/home/n2/chanwoo/UDC/cvpr_rebuttal/UDC-SIT/benchmarks/ECFNet/experiments/backup/ECFNet-3-0.5/model_final.pth"

# Ours: nres 8 version
# model_path="/home/n2/chanwoo/UDC/cvpr_rebuttal/UDC-SIT/benchmarks/ECFNet/experiments/ECFNet-cvpr_rebuttal-3-0.5/model_final.pth"

# This is original parameter
# model_path=/home/n2/chanwoo/UDC/cvpr_rebuttal/ECFNet/Codes/ckpt/model.pth # original parameter

experiment_name="${name}-ours-n6"

# make dir to save image
mkdir -p $save_dir

python test.py \
    --name $experiment_name \
    --model-path $model_path \
    --test-GT $test_gt \
    --test-input $test_input \
    --channels $channels \
    --norm $norm \
    --num-res 6 # change to 8 if testing nres 8 version