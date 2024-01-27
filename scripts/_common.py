# COMMON IMPORTS
import logging

log = logging.getLogger(__name__)

from rich.traceback import install

install()

log.debug("Importing common modules")

import functools
import itertools
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import hydra
import lightning as L
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.func
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.progress import Progress, track
from torch import Tensor, nn
from tqdm.autonotebook import tqdm
from typing_extensions import TypeAlias

log.debug("Finished importing common modules")
torch.set_float32_matmul_precision('medium')

# COMMON CONSTANTS
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
SRC_DIR = PROJECT_DIR / "src"
CONFIG_DIR = PROJECT_DIR / "config"
CACHE_DIR = PROJECT_DIR / "cache"
TEMPLATE_DIR = PROJECT_DIR / "templates"

sys.path.append(str(PROJECT_DIR))

log.debug(f"{PROJECT_DIR=}")

__all__ = [n for n in globals().keys() if not n.startswith("_") and n not in ["log", "install"]]
