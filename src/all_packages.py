import argparse
import datetime
import gc
import json
import os
import os.path as osp
import random
import sys
import time
from datetime import datetime, timedelta

import dotenv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import nn
from tqdm import tqdm

dotenv.load_dotenv(override=True)
DATA_DIR = os.getenv("DATA")
RES_DIR = os.getenv("RES")

## Check path to data

if DATA_DIR is None:
    logger.error("env var DATA not specified in .env")
    sys.exit(1)
if not osp.isdir(DATA_DIR):
    logger.error("DATA dir specified in .env not existed")
    sys.exit(1)


PATHS = {
    "bert": osp.join(RES_DIR, "bert-base-uncased"),
    "rel_info": osp.join(DATA_DIR, "raw", "rel_info.json"),
    "raw_train": osp.join(DATA_DIR, "raw", "train_annotated.json"),
    "rel_entity_dedicated": osp.join(
        DATA_DIR, "prepro", "prepro_data_bert/rel_entity_dedicated.json"
    ),
    "rel2id": osp.join(DATA_DIR, "prepro", "prepro_data_bert/rel2id.json"),
}

MAX_NODE_NUM = 512
IGNORE_INDEX = -100


class NaNReporter:
    EPS = 1e4

    @staticmethod
    def check_NaN(loss: torch.Tensor) -> bool:
        if torch.sum(torch.isnan(loss)):
            logger.warning("Loss is NaN")
            return True

        return False

    @staticmethod
    def check_abnormal(tensor: torch.Tensor, name: str) -> bool:
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
