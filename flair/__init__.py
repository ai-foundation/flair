import os
import subprocess
from io import StringIO

import numpy as np
import pandas as pd
import torch


# from .available import get_free_gpus


def get_avail_device() -> torch.device:
    # gpu_id = get_free_gpus()[0]  # select the first gpu
    gpu_id = get_freer_gpu()
    return torch.device(f"cuda:{gpu_id}")


def get_freer_gpu():
    # https: // discuss.pytorch.org / t / it - there - anyway - to - let - program - select - free - gpu - automatically / 17560 / 2
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def get_free_gpu():
    # https: // discuss.pytorch.org / t / it - there - anyway - to - let - program - select - free - gpu - automatically / 17560 / 2
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(u"".join(gpu_stats)),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(
        lambda x: x.rstrip(' [MiB]'))
    idx = gpu_df['memory.free'].idxmax()
    print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx][
        'memory.free']))
    return idx


# global variable: cache_root
cache_root = os.path.expanduser(os.path.join("~", ".flair"))

# global variable: device
device = None
if torch.cuda.is_available():
    device = get_avail_device()
else:
    device = torch.device("cpu")

from . import data
from . import models
from . import visual
from . import trainers
from . import nn

import logging.config

__version__ = "0.4.2"

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": "%(asctime)-15s %(message)s"}},
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "flair": {"handlers": ["console"], "level": "INFO",
                      "propagate": False}
        },
        "root": {"handlers": ["console"], "level": "WARNING"},
    }
)

logger = logging.getLogger("flair")
