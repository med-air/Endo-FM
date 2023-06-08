#!/bin/bash

EXP_NAME="polypdiag_finetune"
DATASET="ucf101"
DATA_PATH="data/downstream/PolypDiag"
CHECKPOINT="checkpoints/endo_fm.pth"


if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi


python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  eval_finetune.py \
  --n_last_blocks 1 \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --epochs 20 \
  --lr 0.001 \
  --batch_size_per_gpu 4 \
  --num_workers 4 \
  --num_labels 2 \
  --dataset "$DATASET" \
  --output_dir "checkpoints/eval/$EXP_NAME" \
  --test \
  --pretrained_model_weights 'polypdiag.pth' \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}/splits" \
  DATA.PATH_PREFIX "${DATA_PATH}/videos" \
  DATA.USE_FLOW False
