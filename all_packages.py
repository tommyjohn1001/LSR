import argparse
import datetime
import gc
import json
import os
import os.path as osp
import pickle
import random
from collections import defaultdict
from datetime import datetime, timedelta
from glob import glob

import dotenv
import numpy as np
import torch
import torch.nn as nn
import yaml
from loguru import logger
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

DOCRED = "docred"
data_set = DOCRED

BATCH_SIZE = 8
HIDDEN_DIM = 120  # please use 216 for BERT
# for BERT---
LR = 1e-3
MAX_EPOCH = 200

SEED = 1337
NAME = "Struct"
EMB_DIM = 100
DECAY_RATE = 0.98

TRAIN_PREFIX = "dev_train"
TEST_PREFIX = "dev_dev"

IGNORE_INDEX = -100

NO_RELATE = 0
INTRA_COREF = 1
INTRA_RELATE = 2
INTER_COREF = 3
INTER_RELATE = 4
INTRA_ENT_TOK = 5

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

POS_TO_ID = {
    PAD_TOKEN: 0,
    UNK_TOKEN: 1,
    "NNP": 2,
    "NN": 3,
    "IN": 4,
    "DT": 5,
    ",": 6,
    "JJ": 7,
    "NNS": 8,
    "VBD": 9,
    "CD": 10,
    "CC": 11,
    ".": 12,
    "RB": 13,
    "VBN": 14,
    "PRP": 15,
    "TO": 16,
    "VB": 17,
    "VBG": 18,
    "VBZ": 19,
    "PRP$": 20,
    ":": 21,
    "POS": 22,
    "''": 23,
    "``": 24,
    "-RRB-": 25,
    "-LRB-": 26,
    "VBP": 27,
    "MD": 28,
    "NNPS": 29,
    "WP": 30,
    "WDT": 31,
    "WRB": 32,
    "RP": 33,
    "JJR": 34,
    "JJS": 35,
    "$": 36,
    "FW": 37,
    "RBR": 38,
    "SYM": 39,
    "EX": 40,
    "RBS": 41,
    "WP$": 42,
    "PDT": 43,
    "LS": 44,
    "UH": 45,
    "#": 46,
}

MAX_SENT_LEN = 200
MAX_NODE_NUM = 200
MAX_ENTITY_NUM = 100
MAX_SENT_NUM = 30
MAX_NODE_PER_SENT = 40
