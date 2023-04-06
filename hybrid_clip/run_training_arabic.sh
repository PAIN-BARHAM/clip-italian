#!/bin/bash

SCRIPT_DIR=.
MODEL_DIR=modelallcaptions

IMAGE_ENCODER="openai/clip-vit-base-patch32"
TEXT_ENCODER="aubmindlab/bert-large-arabertv2"


python ${SCRIPT_DIR}/run_hybrid_clip.py \
    --output_dir ${MODEL_DIR} \
    --overwrite_output_dir \
    --tokenizer_name=${TEXT_ENCODER} \
    --train_file="/home/think3/Desktop/training_CLIP/MSCOCO_DATASET_AR_1/train_dataset.json" \
    --validation_file="/home/think3/Desktop/training_CLIP/MSCOCO_DATASET_AR_1/valid_dataset.json" \
    --do_train --do_eval \
    --num_train_epochs="80" --max_seq_length 128 \
    --per_device_train_batch_size="128" \
    --per_device_eval_batch_size="128" \
    --learning_rate="0.00008" --warmup_ratio 0.1 --weight_decay 0.0 \
    --preprocessing_num_workers 32 \
    --log_comet \
    --eval_when 5 \
    --text_model_name_or_path=${TEXT_ENCODER} \
    --vision_model_name_or_path=${IMAGE_ENCODER} \
    --run_from_checkpoint /home/think3/Desktop/training_CLIP/model1

