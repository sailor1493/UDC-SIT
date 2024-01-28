#!/bin/bash

name="Feng" # dataset name
channels=3 # feng dataset has 3 channels
norm="tonemap" # feng dataset use tone mapping for normalization

test_gt="/shared/s1/lab08/udc/Feng-S/test/GT"
test_input="/shared/s1/lab08/udc/Feng-S/test/input"
save_dir="results/${name}"
model_path="/shared/s1/lab08/udc/CVPR/pretrained/Feng-S/SRGAN/SRGAN-Gmodel.pth"

experiment_name="${name}_test"

# make dir to save image
mkdir -p $save_dir

python test.py \
    --name $experiment_name \
    --model-path $model_path \
    --test-GT $test_gt \
    --test-input $test_input \
    --channels $channels \
    --norm $norm \
    --data-itself
