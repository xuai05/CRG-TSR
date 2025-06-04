from __future__ import print_function, division

import os
import random
import ctypes
import setproctitle
import time
import sys
import json

import numpy as np
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from utils import command_parser
from utils.misc_util import Logger

from utils.class_finder import model_class, agent_class, optimizer_class
from utils.model_util import ScalarMeanTracker
from utils.data_utils import loading_scene_list
from main_eval import main_eval
from full_eval import full_eval
from single_eval import single_eval

from runners import a3c_train, a3c_val
from tabulate import tabulate
os.environ['CUDA_VISIBLE_DEVICES']="2,3"      



def main():
    setproctitle.setproctitle("Training Relationship Graph totally ")
    args = command_parser.parse_arguments()

    start_time_str = time.strftime(
        '%Y-%m-%d_%H-%M-%S', time.localtime(time.time())
    )
    work_dir = os.path.join(args.work_dir, '{}_{}_{}'.format(args.title, args.phase, start_time_str))
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    log_file = os.path.join(work_dir, 'train.txt')
    sys.stdout = Logger(log_file, sys.stdout)
    sys.stderr = Logger(log_file, sys.stderr)

    tb_log_dir = os.path.join(args.work_dir, 'runs', '{}_{}_{}'.format(args.title, args.phase, start_time_str))
    log_writer = SummaryWriter(log_dir=tb_log_dir)

    save_model_dir = os.path.join(work_dir, 'trained_models')
    args.save_model_dir = save_model_dir
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # start training preparation steps
    print('Training started from: {}'.format(
        time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
    )

    args.num_steps = 50
    target = a3c_val if args.eval else a3c_train

    args.data_dir = os.path.expanduser('~/Data/AI2Thor_offline_data_2.0.2/')
    scenes = loading_scene_list(args)

    if args.detection_feature_file_name is None:
        args.detection_feature_file_name = 'detr_features_{}cls.hdf5'.format(args.num_category)

    print(args)
    create_shared_model = model_class(args.model)
    init_agent = agent_class(args.agent_type)
    optimizer_type = optimizer_class(args.optimizer)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.eval:
        main_eval(args, create_shared_model, init_agent)
        return

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn",force=True)

    shared_model = create_shared_model(args)

    train_total_ep = 0
    n_frames = 0

    if args.pretrained_trans is not None:
        saved_state = torch.load(
            args.pretrained_trans, map_location=lambda storage, loc: storage
        )
        model_dict = shared_model.state_dict()
        
        # for k in saved_state['model'].items():
        #     print(k)
        pretrained_dict = {k: v for k, v in saved_state['model'].items() if
                           (k in model_dict and v.shape == model_dict[k].shape)}
        # for k in pretrained_dict:
        #     print(k)
        # print(type(pretrained_dict))
        model_dict.update(pretrained_dict)
        shared_model.load_state_dict(model_dict)
    
 

    shared_model.share_memory()

    if args.pretrained_trans is not None:
        optimizer = optimizer_type(
            [
                {'params': [v for k, v in shared_model.named_parameters() if
                            v.requires_grad and (k in pretrained_dict)],
                 'lr': args.pretrained_lr},
                {'params': [v for k, v in shared_model.named_parameters() if
                            v.requires_grad and (k not in pretrained_dict)],
                 'lr': args.lr},
            ]
        )
    else:
        optimizer = optimizer_type(
            [v for k, v in shared_model.named_parameters() if v.requires_grad], lr=args.lr
        )

    if args.continue_training is not None:
        saved_state = torch.load(
            args.continue_training, map_location=lambda storage, loc: storage
        )
        shared_model.load_state_dict(saved_state['model'])
        optimizer.load_state_dict(saved_state['optimizer'])

        train_total_ep = saved_state['episodes']
        n_frames = saved_state['frames']

    optimizer.share_memory()

    processes = []

    end_flag = mp.Value(ctypes.c_bool, False)
    manager = mp.Manager()
    train_res_queue = manager.Queue()
    for rank in range(0, args.workers):
        p = mp.Process(
            target=target,
            args=(
                rank,
                args,
                create_shared_model,
                shared_model,
                init_agent,
                optimizer,
                train_res_queue,
                end_flag,
                scenes,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)

    print("Train agents created.")

    train_thin = args.train_thin
    train_scalars = ScalarMeanTracker()

    start_time = time.time()

    lr = args.lr
    best_model_on_val = None
    best_performance_on_val = 0.0
    start = time.time()
    try:
        while train_total_ep < args.max_ep:
            while train_res_queue.empty():
                time.sleep(0.1)
            train_result = train_res_queue.get()
            train_scalars.add_scalars(train_result)
            train_total_ep += 1
            n_frames += train_result['ep_length']
            if train_total_ep % 1000 == 0:
                end = time.time()
                print(str(train_total_ep) + ' episode cost time: ',end - start)
                start = end

            if (args.lr_drop_eps is not None) and (train_total_ep % args.lr_drop_eps == 0) and (lr > args.lr_min):
                lr = lr * args.lr_drop_weight
                if lr < args.lr_min:
                    lr = args.lr_min
                optimizer.param_groups[1]['lr'] = lr

            if (train_total_ep % train_thin) == 0:
                log_writer.add_scalar('n_frames', n_frames, train_total_ep)
                tracked_means = train_scalars.pop_and_reset()
                for k in tracked_means:
                    log_writer.add_scalar(
                        k + '/train', tracked_means[k], train_total_ep
                    )

            if (train_total_ep % args.ep_save_freq) == 0:
                print('{}: {}: {}'.format(
                    train_total_ep, n_frames, time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
                )
                state = {
                    'model': shared_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'episodes': train_total_ep,
                    'frames': n_frames,
                }
                save_path = os.path.join(
                    save_model_dir,
                    '{0}_{1}_{2}_{3}.dat'.format(
                        args.title, n_frames, train_total_ep, start_time_str
                    ),
                )
                torch.save(state, save_path)
                #validate the model just saved
                if args.test_after_train:
                    end_flag.value = True
                    while not train_res_queue.empty():
                        train_res_queue.get()
                    for p in processes:
                        # p.terminate()
                        time.sleep(0.1)
                        p.join()
                    
                    processes = []
                    result_ep, model_ep = single_eval(args, work_dir)
                    if result_ep["success"] > best_performance_on_val:
                        best_model_on_val = model_ep
                        best_performance_on_val = result_ep["success"]
                    print(
                            tabulate(
                                [
                                    ["SPL >= 1:", result_ep["GreaterThan/1/spl"]],
                                    ["Success >= 1:", result_ep["GreaterThan/1/success"]],
                                    ["SPL >= 5:", result_ep["GreaterThan/5/spl"]],
                                    ["Success >= 5:", result_ep["GreaterThan/5/success"]],
                                ],
                                headers=["Metric", "Result"],
                                tablefmt="orgtbl",
                            )
                        )
                    if train_total_ep < args.max_ep:                    
                        end_flag.value = False
                        # print(args)
                        # train_res_queue = mp.Queue()
                        # shared_model.zero_grad()
                        # optimizer.zero_grad()
                        for rank in range(0, args.workers):
                            p = mp.Process(
                                target=target,
                                args=(
                                    rank,
                                    args,
                                    create_shared_model,
                                    shared_model,
                                    init_agent,
                                    optimizer,
                                    train_res_queue,
                                    end_flag,
                                    scenes,
                                    # replay_buffer,
                                ),
                            )
                            p.start()
                            processes.append(p)
                            time.sleep(0.1)

            # if args.test_speed and train_total_ep % 10000 == 0:
            #     print('{} ep/s'.format(10000 / (time.time() - start_time)))
            #     start_time = time.time()

    finally:
        log_writer.close()
        end_flag.value = True
        for p in processes:
            time.sleep(0.1)
            p.join()
    
    print('Training ended from: {}'.format(
        time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
    )

    if args.test_after_train:
        #singel eval
        args.phase = 'eval'
        args.test_or_val = "test"
        args.load_model = best_model_on_val
        filename = 'result.json' + '_' + args.load_model.split('_')[-3]
        args.results_json = os.path.join(work_dir, filename)
        main_eval(args, create_shared_model, init_agent)
        # single_eval(args, work_dir)
        with open(args.results_json, "r") as f:
            result_ep = json.load(f)
        print(
                tabulate(
                    [
                        ["SPL >= 1:", result_ep["GreaterThan/1/spl"]],
                        ["Success >= 1:", result_ep["GreaterThan/1/success"]],
                        ["SPL >= 5:", result_ep["GreaterThan/5/spl"]],
                        ["Success >= 5:", result_ep["GreaterThan/5/success"]],
                    ],
                    headers=["Metric", "Result"],
                    tablefmt="orgtbl",
                )
            )

if __name__ == "__main__":
    main()
