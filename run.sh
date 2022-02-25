#!/bin/bash

cd src

# 1. For GloVe
## Gendata
python gen_data.py 

## Train
python train.py\
    --appdx rel_restrict_v1\
    --wandb

# 2. For BERT
## Gendata
python gen_data_bert.py 

## Train
python train.py\
    --model_name LSR_bert\
    --appdx rel_restrict_v1\
    --wandb