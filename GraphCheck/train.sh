#!/bin/bash

# config-Llama3.3 70B
python train.py \
    --project GraphCheck_Llama3.3_70B \
    --train_dataset MiniCheck_Train \
    --llm_model_name llama_70b \

# config-Qwen2.5 72B
# python train.py \
#     --project GraphCheck_Qwen2.5_72B \
#     --train_dataset MiniCheck_Train \
#     --llm_model_name qwen_72b \
