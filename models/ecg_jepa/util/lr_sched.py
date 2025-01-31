# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math

def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if config['lr'] is None:
        config['lr'] = config['blr'] * config['dataloader']['batch_size'] / 256

    if epoch < config['warmup_epochs']:
        lr = config['lr'] * epoch / config['warmup_epochs']
    else:
        lr = config['min_lr'] + (config['lr'] - config['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config['warmup_epochs']) / (config['epochs'] - config['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
