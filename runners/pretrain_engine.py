""" Borrowed from VTNet_model """
import math
import os
import sys
import h5py
import time
import json
from typing import Iterable

import torch

import utils.misc as utils


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        max_norm: float = 0,
        print_freq: int = 100,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = print_freq
    epoch_time = time.time()

    for i, (global_feature, local_feature, targets) in enumerate(data_loader):

        global_feature_list = global_feature.to(device)
        Traject_Graph_node_input = local_feature.to(device)
        # local_feature = {k: v.type(torch.FloatTensor).to(device) for k, v in local_feature.items() if
        #                  (k != 'locations') and (k != 'targets')}
        targets = targets.to(device)

        outputs = model(global_feature_list, Traject_Graph_node_input)
        losses = criterion(outputs['action'], targets)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if i % print_freq == 0:
            print('Epoch: {:4d}, iter: {:7d} / {:7d}, loss: {:.5f}, cost: {:.2f}'.format(
                epoch, i, len(data_loader), losses, time.time() - epoch_time
            ))
            epoch_time = time.time()


@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        device: torch.device,
        epoch: int,
):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    total = 0
    correct = 0

    correct_prediction = []


    for i, (global_feature, local_feature, targets) in enumerate(data_loader):
        # locations = local_feature['locations']
        # episode_targets = local_feature['targets']
        # idxs = local_feature['idx']

        global_feature = global_feature.to(device)
        Traject_Graph_node_input = local_feature.to(device)
        # local_feature = {k: v.type(torch.FloatTensor).to(device) for k, v in local_feature.items() if
        #                  (k != 'locations') and (k != 'targets')}
        targets = targets.to(device)

        outputs = model(global_feature, Traject_Graph_node_input)
        loss_dict = criterion(outputs['action'], targets)
        _, predicted = torch.max(outputs['action'].data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        correct_prediction.extend(predicted[predicted == targets].tolist())


    with open('./temp.json', 'w') as wf:
        json.dump(correct_prediction, wf)

    print('Epoch: {} accuracy: {}'.format(epoch, correct / total))
