import os
from abc import abstractmethod

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, IterableDataset

from src.datasets.catalog import DATASET_DICT
from src.models import transformer
from src.utils import LABEL_FRACS


def get_model(config: DictConfig, dataset_class: Dataset):
    '''Retrieves the specified model class, given the dataset class.'''
    if config.model.name == 'transformer':
        model_class = transformer.DomainAgnosticTransformer
        # Retrieve the dataset-specific params.
        return model_class(
            input_specs=dataset_class.spec(),
            **config.model.kwargs,
        )
    elif config.model.name == 'imagenet-vit':
        model_class = transformer.ImageNetVisionTransformer
        # No input specs passed to embedding module, only number of input channels.
        return model_class([], dataset_class.IN_CHANNELS)
    else:
        raise ValueError(f'Encoder {config.model.name} doesn\'t exist.')


class BaseSystem(pl.LightningModule):

    def __init__(self, config: DictConfig):
        '''An abstract class that implements some shared functionality for training.
        Args:
            config: a hydra config
        '''
        super().__init__()
        self.config = config
        self.dataset = DATASET_DICT[config.dataset.name]
        self.model = get_model(config, self.dataset)
        if 'finetune_size' in self.config:
            self.finetune_size = 0 if self.config.finetune_size is None else LABEL_FRACS[self.config.finetune_size]

    @abstractmethod
    def objective(self, *args):
        '''Computes the loss and accuracy.'''
        pass

    @abstractmethod
    def forward(self, batch):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    def setup(self, stage):
        '''Called right after downloading data and before fitting model, initializes datasets with splits.'''
        train_ds_kwargs = {"base_root": self.config.data_root, "download": True, "train": True}
        if 'finetune_size' in self.config:
            if self.config.finetune_size in ['small', 'medium', 'large', 'full']:
                train_ds_kwargs["finetune_size"] = self.config.finetune_size
            elif self.config.finetune_size is not None:
                raise ValueError("finetune_size must be one of 'small', 'medium', 'large', 'full', or 'null'.")
        self.train_dataset = self.dataset(**train_ds_kwargs)
        self.val_dataset = self.dataset(base_root=self.config.data_root, download=True, train=False)
        self.batch_size = self.config.dataset.batch_size
        if 'finetune_size' in self.config and self.config.finetune_size and self.finetune_size * self.dataset.NUM_CLASSES < self.batch_size:
            self.batch_size = self.finetune_size * self.dataset.NUM_CLASSES
            print(f"Resetting batch size to {self.batch_size} since finetune_size * num_classes < batch_size")
        try:
            print(f'{len(self.train_dataset)} train examples, {len(self.val_dataset)} val examples')
        except TypeError:
            print('Iterable/streaming dataset- undetermined length.')

    def train_dataloader(self):
        print("batch size", self.batch_size)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.config.dataset.num_workers,
            shuffle=not isinstance(self.train_dataset, IterableDataset),
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        print("batch size", self.batch_size)
        if not self.val_dataset:
            raise ValueError('Cannot get validation data for this dataset')
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.config.dataset.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        print("batch size", self.batch_size)
        if not self.val_dataset:
            raise ValueError('Cannot get validation data for this dataset')
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.config.dataset.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if self.config.optim.name == 'adam':
            optim = torch.optim.AdamW(params, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.name == 'sgd':
            optim = torch.optim.SGD(
                params,
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                momentum=self.config.optim.momentum,
            )
        elif self.config.optim.name == 'adam-warmup-decay':
            import math
            dataset_len = len(self.train_dataset)
            if 'max_steps' in self.config.trainer:
                steps_per_epoch = math.ceil(dataset_len // self.batch_size)
                print("steps", steps_per_epoch)
                epochs=self.config.trainer.max_steps // steps_per_epoch
                print("epochs", epochs)
                print("schedule1", epochs//10)
            else:
                epochs=self.config.trainer.max_epochs

            optim = torch.optim.AdamW(params, lr=self.config.optim.lr*0.1, weight_decay=self.config.optim.weight_decay)
            
            scheduler1 = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1e-30, total_iters=epochs//10)
            scheduler2 = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0., total_iters=epochs-epochs//10)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optim, schedulers=[scheduler1, scheduler2], milestones=[epochs//10])
            return [optim], [scheduler]
        else:
            raise ValueError(f'{self.config.optim.name} optimizer unrecognized.')
        return optim

    def on_train_end(self):
        model_path = os.path.join(self.trainer.checkpoint_callback.dirpath, 'model.ckpt')
        torch.save(self, model_path, pickle_protocol=4)
        print(f'Pretrained model saved to {model_path}')
