# Authors: Yonghao Song <eeyhsong@gmail.com>
#
# License: BSD (3-clause)
import warnings

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, Tensor

from braindecode.models import EEGConformer
import numpy as np



eeg_conformer = EEGConformer(
    n_outputs=4,
    n_chans=32,
    n_filters_time=40,
    filter_time_length=25,
    pool_time_length=75,
    pool_time_stride=15,
    drop_prob=0.5,
    att_depth=6,
    att_heads=10,
    att_drop_prob=0.5,
    final_fc_length=2440,
    return_features=False,
    input_window_samples=750,
    add_log_softmax=True
)

# train_args = {
#     'batch_size': 64,
#     'n_epochs': 2000,
#     'lr': 0.001,
#     'b1': 0.05,
#     'b2': 0.999,
# }


# class ConformerTrain:
#     def __init__(self):

batch_size = 8
n_epochs = 2000
lr = 0.0002
b1 = 0.05
b2 = 0.999

criterion_l1 = torch.nn.L1Loss().cuda()
criterion_l2 = torch.nn.MSELoss().cuda()
criterion_cls = torch.nn.CrossEntropyLoss().cuda()


