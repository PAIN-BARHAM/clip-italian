#!/bin/bash

SCRIPT_DIR=.
MODEL_DIR=~/clip-italian

IMAGE_ENCODER="openai/clip-vit-base-patch32"
TEXT_ENCODER="dbmdz/bert-base-italian-xxl-uncased"


python ${SCRIPT_DIR}/run_hybrid_clip.py \
    --output_dir ${MODEL_DIR} \
    --overwrite_output_dir \
    --text_model_name_or_path=${TEXT_ENCODER} \
    --vision_model_name_or_path=${IMAGE_ENCODER} \
    --tokenizer_name=${TEXT_ENCODER} \
    --train_file="../data/train_dataset.json" \
    --validation_file="../data/valid_dataset.json" \
    --do_train --do_eval \
    --num_train_epochs="80" --max_seq_length 96 \
    --per_device_train_batch_size="128" \
    --per_device_eval_batch_size="64" \
    --learning_rate="5e-5" --warmup_steps="526" --weight_decay 0.1 \
    --preprocessing_num_workers 32 \
    --push_to_hub
