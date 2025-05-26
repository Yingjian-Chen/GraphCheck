#!/bin/bash

# config
COMMON_ARGS="\
--project GraphCheck_Llama3.3_70B \
--eval_batch_size 8 \
--llm_model_name llama_70b \
"

# datasets
DATASETS=(
    "AggreFact-CNN"
    "AggreFact-XSum"
    "summeval"
    "ExpertQA"
    "COVID-Fact"
    "SCIFACT"
    "pubhealth"
)

for dataset in "${DATASETS[@]}"
do
    echo "Evaluating on dataset: $dataset"
    python inference.py --dataset_name $dataset $COMMON_ARGS
done
