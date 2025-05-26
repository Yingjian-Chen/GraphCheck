#!/bin/bash

datasets=(
  "AggreFact-CNN"
  "AggreFact-XSum"
  "summeval"
  "ExpertQA"
  "COVID-Fact"
  "SCIFACT"
  "pubhealth"
)

for dataset in "${datasets[@]}"
do
  echo "=== Processing dataset: $dataset ==="
  python graph_build.py --data_name "$dataset"
done