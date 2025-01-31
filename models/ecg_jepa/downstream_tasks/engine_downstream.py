# Original work Copyright (c) Meta Platforms, Inc. and affiliates. <https://github.com/facebookresearch/mae>
# Modified work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Dict, Iterable, Optional, Tuple

import torch
import torchmetrics

import util.misc as misc
import util.lr_sched as lr_sched

import torch
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    config: Optional[dict] = None,
                    use_amp: bool = True,
                    ) -> Dict[str, float]:
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    accum_iter = config['accum_iter']
    max_norm = config['max_norm']

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(samples)
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                targets = targets.to(dtype=outputs.dtype)       

  
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss,
                    optimizer,
                    clip_grad=max_norm,
                    parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((epoch + data_iter_step / len(data_loader)) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader: Iterable,
             device: torch.device,
             metric_fn: torchmetrics.Metric,
             output_act: torch.nn.Module,
             target_dtype: torch.dtype = torch.long,
             use_amp: bool = True,
             ) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if samples.ndim == 4:  # batch_size, n_drops, n_channels, n_frames
                logits_list = []
                for i in range(samples.size(1)):
                    logits = model(samples[:, i])
                    logits_list.append(logits)
                logits_list = torch.stack(logits_list, dim=1)
                outputs_list = output_act(logits_list)
                logits = logits_list.mean(dim=1)
                outputs = outputs_list.mean(dim=1)
            else:
                logits = model(samples)
                outputs = output_act(logits)
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                targets = targets.to(dtype=outputs.dtype)
            loss = criterion(logits, targets)

        outputs = misc.concat_all_gather(outputs)
        targets = misc.concat_all_gather(targets).to(dtype=target_dtype)
        metric_fn.update(outputs, targets)
        metric_logger.meters['loss'].update(loss.item(), n=samples.size(0))

    metric_logger.synchronize_between_processes()
    valid_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    metrics = metric_fn.compute()
    if isinstance(metrics, dict):  # MetricCollection
        metrics = {k: v.item() for k, v in metrics.items()}
    else:
        metrics = {metric_fn.__class__.__name__: metrics.item()}
    metric_str = "  ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
    metric_str = f"{metric_str} loss: {metric_logger.loss.global_avg:.3f}"
    # print(f"* {metric_str}")
    metric_fn.reset()

    return valid_stats, metrics


# @torch.no_grad()
# def evaluate(model: torch.nn.Module,
#              criterion: torch.nn.Module,
#              data_loader: Iterable,
#              device: torch.device,
#              metric_fn: torchmetrics.Metric,
#              output_act: torch.nn.Module,
#              target_dtype: torch.dtype = torch.long,
#              use_amp: bool = True,
#              ) -> Tuple[Dict[str, float], Dict[str, float]]:
#     model.eval()
#     metric_logger = misc.MetricLogger(delimiter="  ")
#     header = 'Test:'
#     label_map = {5: 'BBB', 11: 'PVC', 13: 'Paced', 4: 'AVB4', 0: 'AFIB', 14: 'SA',
#                  6: 'BBB_AFIB', 12: 'PVC2', 3: 'AVB2', 1: 'AFL', 8: 'NSR', 15: 'VT',
#                  2: 'AVB1', 9: 'PAC', 10: 'PAC2', 7: 'NQT'}
#     all_targets = []
#     all_predictions = []

#     for samples, targets in metric_logger.log_every(data_loader, 50, header):
#         samples = samples.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)

#         with torch.cuda.amp.autocast(enabled=use_amp):
#             if samples.ndim == 4:  # batch_size, n_drops, n_channels, n_frames
#                 logits_list = []
#                 for i in range(samples.size(1)):
#                     logits = model(samples[:, i])
#                     logits_list.append(logits)
#                 logits_list = torch.stack(logits_list, dim=1)
#                 outputs_list = output_act(logits_list)
#                 logits = logits_list.mean(dim=1)
#                 outputs = outputs_list.mean(dim=1)
#             else:
#                 logits = model(samples)
#                 outputs = output_act(logits)
#             if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
#                 targets = targets.to(dtype=outputs.dtype)
#             loss = criterion(logits, targets)

#         outputs = misc.concat_all_gather(outputs)
#         targets = misc.concat_all_gather(targets).to(dtype=target_dtype)
#         metric_fn.update(outputs, targets)
#         metric_logger.meters['loss'].update(loss.item(), n=samples.size(0))

#         # Save all targets and predictions for confusion matrix and accuracy calculation
#         all_targets.append(targets.cpu().numpy())
#         all_predictions.append(torch.argmax(outputs, dim=1).cpu().numpy())

#     # Concatenate all targets and predictions
#     all_targets = np.concatenate(all_targets)
#     all_predictions = np.concatenate(all_predictions)

#     # Calculate accuracy
#     accuracy = accuracy_score(all_targets, all_predictions)

#     # Calculate confusion matrix
#     conf_matrix = confusion_matrix(all_targets, all_predictions)

#     # Print accuracy
#     print(f'Accuracy: {accuracy:.3f}')

#     # Print and save confusion matrix
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
#                 xticklabels=[label_map[i] for i in range(len(label_map))], 
#                 yticklabels=[label_map[i] for i in range(len(label_map))])
#     plt.title('Confusion Matrix')
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.savefig('confusion_matrix.png')
#     plt.close()

#     metric_logger.synchronize_between_processes()
#     valid_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

#     metrics = metric_fn.compute()
#     if isinstance(metrics, dict):  # MetricCollection
#         metrics = {k: v.item() for k, v in metrics.items()}
#     else:
#         metrics = {metric_fn.__class__.__name__: metrics.item()}
#     metric_str = "  ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
#     metric_str = f"{metric_str} loss: {metric_logger.loss.global_avg:.3f}"
#     # print(f"* {metric_str}")
#     metric_fn.reset()

#     return valid_stats, metrics
