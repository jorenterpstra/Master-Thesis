# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

import timm
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import os
import numpy as np

from utils import timer


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # Add accuracy metrics for training
    metric_logger.add_meter('train_acc1', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    metric_logger.add_meter('train_acc5', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    # Check if dataset includes custom rankings - directly use the flag from args
    has_rankings = args.has_rankings.get('train', False)
    
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        # Unpack batch based on whether it includes rankings
        if has_rankings:
            samples, targets, rankings = batch
        else:
            samples, targets = batch
            rankings = None
            
        if args.debug:
            print("------------- Samples are being loaded to device ", device)
            
        with timer("Loading samples to device"):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
        if rankings is not None:
            rankings = rankings.to(device, non_blocking=True)
            # Handle global ordering case
            if rankings.dim() == 1:  # Shape [P]
                B = samples.shape[0]  # Get batch size
                rankings = rankings.unsqueeze(0).expand(B, -1)  # Expand to [B, P]
                if args.debug:
                    print(f"Expanded global ranking of shape {rankings.shape}")
            
        if args.debug:
            print(f'------------- Samples are now loaded on device {samples.device}' )
            print(f"------------- The target is now loaded on device {targets.device}")
            if rankings is not None:
                print(f"------------- The rankings are now loaded on device {rankings.device}")

        original_targets = targets.clone()  # Save original targets for accuracy calculation
        
        if mixup_fn is not None:
            # print('Mixing up samples')
            samples, targets = mixup_fn(samples, targets)
            # print('Samples mixed up')
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            if rankings is not None:
                # In case of cosub, also duplicate the rankings
                rankings = torch.cat((rankings, rankings), dim=0)
        # print('Samples concatenated')

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
        # print('BCE loss applied')

        with amp_autocast():
            with timer("Forward pass"):
                outputs = model(
                    samples, 
                    if_random_cls_token_position=args.if_random_cls_token_position, 
                    if_random_token_rank=args.if_random_token_rank,
                    custom_rank=rankings
                )
            
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        if args.if_nan2num:
            with amp_autocast():
                with timer("Loss scaling"):
                    loss = torch.nan_to_num(loss)
        loss_value = loss.item()
        if args.debug:
            print('Loss value calculated = ', loss_value)
        
        # Calculate training accuracy (only when not using mixup or cosub to ensure valid metrics)
        if not args.cosub and mixup_fn is None:
            acc1, acc5 = accuracy(outputs, original_targets, topk=(1, 5))
            metric_logger.meters['train_acc1'].update(acc1.item(), n=samples.shape[0])
            metric_logger.meters['train_acc5'].update(acc5.item(), n=samples.shape[0])
        elif not args.cosub:
            # If using mixup, we need a rough accuracy estimate
            # This won't be totally accurate but gives a trend
            with torch.no_grad():
                # Get predictions on the original data
                pred_outputs = model(samples)
                acc1, acc5 = accuracy(pred_outputs, original_targets, topk=(1, 5))
                metric_logger.meters['train_acc1'].update(acc1.item(), n=samples.shape[0])
                metric_logger.meters['train_acc5'].update(acc5.item(), n=samples.shape[0])

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)

        optimizer.zero_grad()
        if args.debug:
            print('Optimizer zero grad done')

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
            if args.debug:
                print('Loss scaled')
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            if args.debug:
                print('Optimizer step done')
        if args.debug:
            print('Model training done, waiting for GPU synchronization')
        torch.cuda.synchronize()
        if args.debug:
            print('GPU synchronized')
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    if args.debug:
        print('Synchronizing between processes')
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast, save_predictions=False, output_dir=None, args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    
    # Use the has_rankings flag directly from args
    has_rankings = args.has_rankings.get('val', False) if args is not None and hasattr(args, 'has_rankings') else False
    
    # Lists to store predictions and labels
    all_predictions = []
    all_targets = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        # Unpack batch based on whether it includes rankings
        if has_rankings:
            images, target, rankings = batch
        else:
            images, target = batch
            rankings = None
            
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if rankings is not None:
            rankings = rankings.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(images, custom_rank=rankings)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        # Store predictions and true labels
        predictions = output.argmax(dim=1).cpu()
        all_predictions.append(predictions)
        all_targets.append(target.cpu())

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    # Convert list of tensors to single tensors
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Save predictions and targets if requested
    if save_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'predictions.npy'), all_predictions.numpy())
        np.save(os.path.join(output_dir, 'targets.npy'), all_targets.numpy())
        print(f"Saved predictions and targets to {output_dir}")
    
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['predictions'] = all_predictions
    results['targets'] = all_targets
    
    return results
