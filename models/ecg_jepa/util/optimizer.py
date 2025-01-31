# Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def get_optimizer_from_config(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_name = config['optimizer']
    lr = config['lr']
    weight_decay = config['weight_decay']
    kwargs = config.get('optimizer_kwargs', {})
    
    if opt_name == "sgd":
        momentum = kwargs.get('momentum', 0)
        return torch.optim.SGD(model.parameters(),
                               lr=lr,
                               momentum=momentum,
                               weight_decay=weight_decay)
    
    elif opt_name == "adamw":
        betas = kwargs.get('betas', (0.9, 0.999))
        if isinstance(betas, list):
            betas = tuple(betas)
        eps = kwargs.get('eps', 1e-8)
        return torch.optim.AdamW(model.parameters(),
                                 lr=lr,
                                 betas=betas,
                                 eps=eps,
                                 weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
