import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import logging  # Import the logging module
from datetime import datetime  # Import the datetime module

# Get the current working directory
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from ecg_data import *

from ecg_data import ECGDataset
from torch.utils.data import DataLoader
from models import load_encoder
from linear_probe_utils import FinetuningClassifier
import torch.nn as nn
from augmentation import *


import util.misc as misc
from engine_downstream import evaluate, train_one_epoch
from util.losses import build_loss_fn
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.optimizer import get_optimizer_from_config
from util.perf_metrics import build_metric_fn, is_best_metric
from models import load_encoder

def parse():
    parser = argparse.ArgumentParser('ECG downstream training')

    # parser.add_argument('--model_name',
    #                     default="mvt_larger_larger",
    #                     type=str,
    #                     help='resume from checkpoint')
    
    parser.add_argument('--ckpt_dir',
                        default="../weights/multiblock_epoch100.pth",
                        type=str,
                        metavar='PATH',
                        help='pretrained encoder checkpoint')
    
    parser.add_argument('--output_dir',
                        default="./output/finetuning",
                        type=str,
                        metavar='PATH',
                        help='output directory')
    
    parser.add_argument('--dataset',
                        default="ptbxl",
                        type=str,
                        help='dataset name')
    
    parser.add_argument('--data_dir',
                        default="/mount/ecg/ptb-xl-1.0.3/",
                        type=str,
                        help='dataset directory')
    
    parser.add_argument('--task',
                        default="multiclass",
                        type=str,
                        help='downstream task')

    parser.add_argument('--data_percentage',
                        default=1.0,
                        type=float,
                        help='data percentage (from 0 to 1) to use in few-shot learning')
    


    # Use parse_known_args instead of parse_args
    args, unknown = parser.parse_known_args()


    with open(os.path.realpath(f'../configs/downstream/finetuning/fine_tuning_ejepa.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    for k, v in vars(args).items():
        if v:
            config[k] = v

    return config

def main(config):

    os.makedirs(config['output_dir'], exist_ok=True)
    
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create log filename with current time
    ckpt_name = os.path.splitext(os.path.basename(config['ckpt_dir']))[0]
    log_filename = os.path.join(config['output_dir'], 
                                f'log_{ckpt_name}_{config["task"]}_{config["dataset"]}_{current_time}.txt')
    
    # Configure logging
    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Log the config dictionary
    logging.info("Configuration:")
    logging.info(yaml.dump(config, default_flow_style=False))

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    task = config['task']
    config['metric']['task'] = task

    # define data augmentation
    aug = {
    'rand_augment': {'use': True, 'kwargs': {'op_names': ['shift', 'cutout', 'drop', 'flip', 'erase', 'sine', 'partial_sine', 'partial_white_noise'], 'level': 10, 'num_layers': 2, 'prob': 0.5}}, 
    'train_transforms': [{'highpass_filter': {'fs': 250, 'cutoff': 0.67}}, {'lowpass_filter': {'fs': 250, 'cutoff': 40}} ], 
    'eval_transforms': [{'highpass_filter': {'fs': 250, 'cutoff': 0.67}}, {'lowpass_filter': {'fs': 250, 'cutoff': 40}} ]
    }

    # # st_mem model requires shorter input length
    # if config['model_name'] == 'st_mem':
    #     aug['train_transforms'].append({'random_crop': {'crop_length': 2250}})
    #     aug['eval_transforms'].append({'random_crop': {'crop_length': 2250}})

    train_transforms = get_transforms_from_config(aug["train_transforms"])
    randaug_config = aug.get("rand_augment", {})
    use_randaug = randaug_config.get("use", False)
    if use_randaug:
        randaug_kwargs = randaug_config.get("kwargs", {})
        train_transforms.append(get_rand_augment_from_config(randaug_kwargs))

    test_transforms = get_transforms_from_config(aug["eval_transforms"])

    train_transforms = Compose(train_transforms + [ToTensor()])
    test_transforms = Compose(test_transforms + [ToTensor()])

    # load dataset
    logging.info(f'Loading {config["dataset"]} dataset...')
    print(f'Loading {config["dataset"]} dataset...')
    waves_train, waves_test, labels_train, labels_test = waves_from_config(config)
    
    if task == 'multilabel':
        _, n_labels = labels_train.shape
        config['metric']['num_labels'] = n_labels
        print(f"Number of labels: {n_labels}")
        n = n_labels
    else:
        n_classes = len(np.unique(labels_train))
        config['metric']['num_classes'] = n_classes
        print(f"Number of classes: {n_classes}")
        n = n_classes
    
    train_dataset = ECGDataset(waves_train, labels_train, train_transforms)
    test_dataset = ECGDataset(waves_test, labels_test, test_transforms)

    data_loader_train = DataLoader(train_dataset, batch_size=config['dataloader']['batch_size'], shuffle=True, num_workers=config['dataloader']['num_workers'])
    data_loader_test = DataLoader(test_dataset, batch_size=config['dataloader']['batch_size'], shuffle=False, num_workers=config['dataloader']['num_workers'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info(f'Loading encoder from {config["ckpt_dir"]}...')
    print(f'Loading encoder from {config["ckpt_dir"]}...')
    encoder, embed_dim = load_encoder(ckpt_dir=config['ckpt_dir'])
    encoder = encoder.to(device)
    model = FinetuningClassifier(encoder, encoder_dim=embed_dim, num_labels=n).to(device)

    lr = config['train']['blr'] * config['dataloader']['batch_size']/256
    config['train']['lr'] = lr
    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=lr,
                                betas=(0.9, 0.95),
                                weight_decay=config['train']['weight_decay'])

    loss_scaler = NativeScaler()
    criterion = nn.BCEWithLogitsLoss() if config['task'] == 'multilabel' else nn.CrossEntropyLoss()
    output_act = nn.Sigmoid() if config['task'] == 'multilabel' else nn.Softmax(dim=-1)
    best_loss = float('inf')

    metric_fn, best_metrics = build_metric_fn(config['metric'])
    metric_fn.to(device)

    import time

    # Start training
    start_time = time.time()
    use_amp = True

    output_dir = config['output_dir']
    log_writer = None

    for epoch in range(config['train']['epochs']):
        train_stats = train_one_epoch(model,
                                        criterion,
                                        data_loader_train,
                                        optimizer,
                                        device,
                                        epoch,
                                        loss_scaler,
                                        log_writer,
                                        config['train'],
                                        use_amp=use_amp,
                                        )

        valid_stats, metrics = evaluate(model,
                                        criterion,
                                        data_loader_test,
                                        device,
                                        metric_fn,
                                        output_act,
                                        use_amp=use_amp,
                                        )
        curr_loss = valid_stats['loss']
        
        # Log epoch metrics
        logging.info(f"Epoch: {epoch}")
        logging.info(f"Training Loss: {train_stats['loss']:.4f}")
        logging.info(f"Validation Loss: {curr_loss:.4f}")

        print(f"Epoch: {epoch}")
        print(f"Training Loss: {train_stats['loss']:.4f}")
        print(f"Validation Loss: {curr_loss:.4f}")

        for metric_name, metric_class in metric_fn.items():
            curr_metric = metrics[metric_name]
            logging.info(f"{metric_name}: {curr_metric:.3f}")
            print(f"{metric_name}: {curr_metric:.3f}")
            if is_best_metric(metric_class, best_metrics[metric_name], curr_metric):
                best_metrics[metric_name] = curr_metric
            logging.info(f"Best {metric_name}: {best_metrics[metric_name]:.3f}")
            print(f"Best {metric_name}: {best_metrics[metric_name]:.3f}")

        print('========================================================================================')


if __name__ == '__main__':
    config = parse()


    # pretrained_ckpt_dir = {
    #     'ejepa_random': f"../weights/random_epoch100.pth",
    #     'ejepa_multiblock': f"../weights/multiblock_epoch100.pth",
    #     # 'cmsc': "../weights/shao+code15/CMSC/epoch300.pth",
    #     # 'cpc': "../weights/shao+code15/cpc/base_epoch100.pth",
    #     # 'simclr': "../weights/shao+code15/SimCLR/epoch300.pth",
    #     # 'st_mem': "../weights/shao+code15/st_mem/st_mem_vit_base.pth",
    # }
        

    # # pretrained_ckpt_dir['mvt_larger_larger'] = f"../weights/shao+code15/block_masking/jepa_v4_20240720_215455_(0.175, 0.225)/epoch{100}.pth"

    # config['ckpt_dir'] = pretrained_ckpt_dir[config['model_name']]

    main(config)




