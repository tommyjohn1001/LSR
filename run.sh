#!/bin/bash

cd src

# 1. For GloVe
## Gendata
# python gen_data.py 

## Train
# python train.py\
#     --appdx GNN_replace\
#     --wandb

# 2. For BERT
## Gendata
# python gen_data_bert.py 

## Train
python train.py\
    --model_name LSR_bert\
    --appdx GNN_replace_ResGatedGCN\
    --batch_size 10\
    --lr 0.0001\
    --wandb