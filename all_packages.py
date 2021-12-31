import argparse
import datetime
import gc
import json
import os
import os.path as osp
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
