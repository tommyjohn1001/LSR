#!/bin/bash

cd src

# 1. For GloVe
## Gendata
# python gen_data.py 

## Train
# python train.py\
#     --appdx structure_mask\
#     --wandb

# 2. For BERT
## Gendata
# python gen_data_bert.py 

## Train
python train.py\
    --model_name LSR_bert\
    --appdx structure_mask\
    --batch_size 10\
    --wandb