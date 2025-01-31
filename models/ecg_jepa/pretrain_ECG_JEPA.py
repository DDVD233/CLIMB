import os
import logging
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from scipy.signal import resample
import time
from torch.cuda.amp import GradScaler, autocast
from ecg_jepa import ecg_jepa
from timm.scheduler import CosineLRScheduler
from ecg_data import *
import argparse

def downsample_waves(waves, new_size):
    return np.array([resample(wave, new_size, axis=1) for wave in waves])

# Argument parser
parser = argparse.ArgumentParser(description="Pretrain the JEPA model with ECG data")
parser.add_argument('--mask_scale', type=float, nargs=2, default=[0.175, 0.225], help="Scale of masking")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
parser.add_argument('--mask_type', type=str, default='block', help="Type of masking") # 'block' or 'random'
parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
parser.add_argument('--wd', type=float, default=0.05, help="Weight decay")
parser.add_argument('--data_dir_shao', type=str, default='/mount/ecg/physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords/', help="Directory for Shaoxing data")
parser.add_argument('--data_dir_code15', type=str, default='/mount/ecg/code15', help="Directory for Code15 data")

args = parser.parse_args()

# Access the arguments like this
mask_scale = tuple(args.mask_scale)
batch_size = args.batch_size
lr = args.lr
mask_type = args.mask_type
epochs = args.epochs
wd = args.wd
data_dir_shao = args.data_dir_shao
data_dir_code15 = args.data_dir_code15

# Generate timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Create logs directory if it doesn't exist
save_dir = f'./weights/ecg_jepa_{timestamp}_{mask_scale}'
os.makedirs(save_dir, exist_ok=True)
log_file = os.path.join(save_dir, f'training_{timestamp}.log')

# Configure logging
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_params(params_dict):
    for key, value in params_dict.items():
        logging.info(f'{key}: {value}')
        print(f'{key}: {value}')  
        
os.makedirs(save_dir, exist_ok=True)

start_time = time.time()

# Shaoxing (Ningbo + Chapman)
waves_shaoxing = waves_shao(data_dir_shao)
waves_shaoxing = downsample_waves(waves_shaoxing, 2500)
print(f'Shao waves shape: {waves_shaoxing.shape}')
logging.info(f'Shao waves shape: {waves_shaoxing.shape}')

dataset = ECGDataset_pretrain(waves_shaoxing)

# Code15
dataset_code15 = Code15Dataset(data_dir_code15)
print(f'Code15 waves shape: ({len(dataset_code15.file_indices)}, 8, 2500)')
logging.info(f'Code15 waves shape: ({len(dataset_code15.file_indices)}, 8, 2500)')

loading_time = time.time() - start_time
print(f'Data loading time: {loading_time:.2f}s')

dataset = ConcatDataset([dataset, dataset_code15])
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
del waves_shaoxing

model = ecg_jepa(encoder_embed_dim=768, 
                encoder_depth=12, 
                encoder_num_heads=16,
                predictor_embed_dim=384,
                predictor_depth=6,
                predictor_num_heads=12,
                drop_path_rate=0.1,
                mask_scale=mask_scale,
                mask_type=mask_type,
                pos_type='sincos',
                c=8,
                p=50,
                t=50).to('cuda')


total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params_million = total_params / 1000000
logging.info(f'Total number of learnable parameters: {total_params_million:.2f} million')

param_groups = [{'params': (p for n, p in model.named_parameters() if (p.requires_grad) and ('bias' not in n) and (len(p.shape) != 1))},
                {'params': (p for n, p in model.named_parameters() if (p.requires_grad) and (('bias' in n) or (len(p.shape) == 1))), 
                'WD_exclude': True, 
                'weight_decay': 0}]

iterations_per_epoch = len(train_loader)
optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
scheduler = CosineLRScheduler(optimizer,
        t_initial=iterations_per_epoch*epochs,
        cycle_mul=1,
        lr_min=1e-6,
        cycle_decay=0.1,
        warmup_lr_init=1e-6,
        warmup_t=5*iterations_per_epoch,
        cycle_limit=1,
        t_in_epochs=True)

ema = [0.996,1.0]
momentum_target_encoder_scheduler = (ema[0] + i*(ema[1]-ema[0])/(iterations_per_epoch*epochs) for i in range(int(iterations_per_epoch*epochs)+1))

# Log hyperparameters and model arguments
hyperparameters = vars(args)
log_params(hyperparameters)

scaler = GradScaler()

for epoch in range(epochs):
    start_time = time.time()
    model.train()
    total_loss = 0.
    for minibatch, wave in enumerate(train_loader):
        scheduler.step(epoch * iterations_per_epoch + minibatch)
        bs, c, t = wave.shape
        wave = wave.to('cuda')

        optimizer.zero_grad()
        with autocast():  # Enable mixed precision
            loss = model(wave)    

        # Scale the loss and backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()    

        total_loss += loss.item()

        with torch.no_grad():
            m = next(momentum_target_encoder_scheduler)
            for param_q, param_k in zip(model.encoder.parameters(), model.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

    total_loss /= len(train_loader)
    epoch_time = time.time() - start_time
    print(f'epoch={epoch:04d}/{epochs:04d}  loss={total_loss:.4f}  time={epoch_time:.2f}s')
    logging.info(f'epoch={epoch:04d}/{epochs:04d}  loss={total_loss:.4f}  time={epoch_time:.2f}s')

    if epoch > 1 and (epoch + 1) % 5 == 0:
        model.to('cpu')
        torch.save({'encoder': model.encoder.state_dict(),
                    'epoch': epoch,
                    }, f'{save_dir}/epoch{epoch + 1}.pth')
        model.to('cuda')
