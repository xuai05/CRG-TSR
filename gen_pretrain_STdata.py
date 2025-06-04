from __future__ import print_function, division
import os
import json
import time

from utils import command_parser
from utils.class_finder import model_class, agent_class
from main_eval import main_eval
from tqdm import tqdm
from tabulate import tabulate
import os

from utils.data_utils import loading_scene_list

os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.multiprocessing as mp

import time
import numpy as np
import random
import json
from tqdm import tqdm

from utils.model_util import ScalarMeanTracker
from runners import a3c_val
from tensorboardX import SummaryWriter

os.environ["OMP_NUM_THREADS"] = "1"


import time
import torch
import setproctitle
import copy
import numpy as np

from datasets.constants import AI2THOR_TARGET_CLASSES
from datasets.data import name_to_num

from models.model_io import ModelOptions

from runners.train_util import (
    compute_loss,
    new_episode,
    run_episode,
    end_episode,
    reset_player,
    compute_spl,
    get_bucketed_metrics,
)





def den_preSTdata(args=None, train_dir=None):
    if args is None:
        args = command_parser.parse_arguments()

    args.phase = 'eval'
    args.episode_type = 'TestValEpisode'
    args.test_or_val = 'val'

    args.data_dir = os.path.expanduser('~/Data/AI2Thor_offline_data_2.0.2/')

    if args.detection_feature_file_name is None:
        args.detection_feature_file_name = 'detr_features_{}cls.hdf5'.format(args.num_category)

    create_shared_model = model_class(args.model)
    init_agent = agent_class(args.agent_type)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.num_steps = 50
    scenes = loading_scene_list(args)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    model_to_open = "~/test_22817598_1600000_2022-02-02_22-05-07.dat"

    targets = AI2THOR_TARGET_CLASSES[args.num_category]

    create_shared_model = model_class(args.model)
    shared_model = create_shared_model(args)

    if model_to_open != "":
        saved_state = torch.load(
            model_to_open, map_location=lambda storage, loc: storage
        )
        shared_model.load_state_dict(saved_state['model'])

    init_agent = agent_class(args.agent_type)

    for s in scenes:
        player = init_agent(create_shared_model, args, 0, s, targets, gpu_id=1)
        player.sync_with_shared(shared_model)
        count = 0
        model_options = ModelOptions()
        print("player state is {}".format(player.episode.scene))
        while count < 10:
            # Get a new episode.
            print("count is {}".format(count))
            total_reward = 0
            player.eps_len = 0
            new_episode(args, player)
            player_start_state = copy.deepcopy(player.environment.controller.state)
            player_start_time = time.time()

            while not player.done:
                # Make sure model is up to date.
                player.sync_with_shared(shared_model)
                total_reward = run_episode(player, args, total_reward, model_options, False, shared_model)
                # Compute the loss.
                if not player.done:
                    reset_player(player)
                print("player state is {}".format(player.episode.environment.controller.state))

            spl, best_path_length = compute_spl(player, player_start_state)

            bucketed_spl = get_bucketed_metrics(spl, best_path_length, player.success)
        
            count += 1
            reset_player(player)

        player.exit()


if __name__ == "__main__":
    den_preSTdata()