#!/bin/bash
set -u

DATADIR="dataset/HQS/Mask_phrase/run"
OUTPUTDIR="HQS"
BATCH_SIZE=8
NUM_TRAIN_EPOCHS=10

mkdir -p output_dir/mask_expts/${OUTPUTDIR}

python -u infilling_model/train_infill.py --fp16 \
    --lr 1e-5 \
    --transformer_model biobart-v2-large \
    --data_dir ${DATADIR} \
    --batch_size ${BATCH_SIZE} \
    --output_dir output_dir/mask_expts/${OUTPUTDIR} \
    --epochs ${NUM_TRAIN_EPOCHS} \
    --max_input_len 512 \
    --seed 8888 \
    --do_train true \
    --name ${OUTPUTDIR}