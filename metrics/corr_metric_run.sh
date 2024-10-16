#!/bin/bash

# run command is /bin/bash corr_metric_run.sh

export DATASET_NAME="HQS"
export TASK_MODEL_NAME=""
export MODEL_TYPE="baseline"
export CORR_MODEL_TYPE=""
export EVAL_FILENAME=""


# Rouge 1/2/L + BertScore
# Questeval + FaR + SummaC + C F1

python cmp_metric.py \
  --pred_fp "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/$EVAL_FILENAME" \
  --dec_dir "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/dec_dir/" \
  --ref_dir "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/ref_dir/" \
  --test_linked "dataset/$DATASET_NAME/test_linked.jsonl"


# ClinicalBLEURT
python -m bleurt.score_files \
  -input_file "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/$EVAL_FILENAME" \
  -bleurt_checkpoint "" \
  -scores_file "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/results.txt" \

# MedBARTScore
export METRIC_NAME="MedBARTScore"

mkdir -p "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/$METRIC_NAME"

python bartscore_create_medical_weight.py \
  --input "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/$EVAL_FILENAME" \
  --output "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/$METRIC_NAME/" \
  --file "" \
  --checkpoint ""

python create_pkl.py \
  --input "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/$EVAL_FILENAME" \
  --output "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/$METRIC_NAME/"

python score.py \
  --file  "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/$METRIC_NAME/full.pkl"\
  --bart_path "" \
  --output "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/$METRIC_NAME/result.json" \
  --result_path "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/results.txt" \
  --weight \
  --bart_score

# MedBERTScore
export METRIC_NAME="MedBERTScore"

mkdir -p "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/$METRIC_NAME"

python deberta_create_medical_weight.py \
  --input "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/$EVAL_FILENAME" \
  --output "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/$METRIC_NAME/" \
  --file ""

python score_cli.py -s \
  --input "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/$EVAL_FILENAME" \
  -wc "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/$METRIC_NAME/test_weight_cands.pkl" \
  -wr "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/$METRIC_NAME/test_weight_refs.pkl" \
  --output "output/$TASK_MODEL_NAME/$MODEL_TYPE/corr_instruct/$CORR_MODEL_TYPE/results.txt" \
  --model "" \
  --lang en


