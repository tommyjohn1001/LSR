import gc
import json
import os
import os.path as osp
import random
import time
from abc import abstractclassmethod

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

PATHS = {
    "rel_info": "../data/rel_info.json",
    "raw_train": "../data/train_annotated.json",
    "rel_entity_dedicated": "prepro_data_bert/rel_entity_dedicated.json",
    "rel2id": "prepro_data_bert/rel2id.json",
}


class NaNReporter:
    EPS = 1e4

    @abstractclassmethod
    def check_NaN(self, loss: torch.Tensor) -> bool:
        if torch.sum(torch.isnan(loss)):
            logger.warning(f"Loss is NaN")
            return True

        return False

    @abstractclassmethod
    def check_abnormal(self, tensor: torch.Tensor, name: str) -> bool:
        abnormal = False

        ## check max
        if torch.max(tensor) >= NaNReporter.EPS:
            logger.warning(f"Tensor {name} reaches +inf")
            abnormal = True

        ## check min
        if torch.min(tensor) <= -NaNReporter.EPS:
            logger.warning(f"Tensor {name} reaches +inf")
            abnormal = True

        ## check max
        if torch.sum(torch.isnan(tensor)):
            logger.warning(
                f"Tensor {name} reaches NaN: max {torch.max(tensor)} - min {torch.min(tensor)}"
            )
            abnormal = True

        return abnormal
