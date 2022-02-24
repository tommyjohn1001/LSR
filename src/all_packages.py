import gc
import json
import os
import os.path as osp
import random
import time

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
