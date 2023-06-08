#!/bin/bash

DATA_PATH="./data/pretrain"
EXP_NAME="foundation_surgical_clips32k"


if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi


python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port="$RANDOM" \
  train_ssl.py \
  --arch "timesformer" \
  --batch_size_per_gpu 3 \
  --data_path "${DATA_PATH}" \
  --output_dir "checkpoints/$EXP_NAME" \
  --opts \
  MODEL.TWO_STREAM False \
  MODEL.TWO_TOKEN False \
  DATA.NO_FLOW_AUG False \
  DATA.USE_FLOW False \
  DATA.RAND_CONV False \
  DATA.NO_SPATIAL False

