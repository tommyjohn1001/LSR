#!/bin/bash

# 1. For GloVe
## Gendata
python gen_data.py 

## Train
python code/train.py\
    --appdx structure_mask\
    --wandb

# 2. For BERT
## Gendata
python gen_data_bert.py 

## Train
python code/train.py --model_name LSR_bert\
    --appdx structure_mask\
    --wandb