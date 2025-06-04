import os
import h5py
import json
import shutil
import filecmp

from torch.multiprocessing import Manager
from tqdm import tqdm
from networkx.readwrite import json_graph

import torch
from PIL import Image


# loading the possible scenes
def loading_scene_list(args):
    scenes = []

    for i in range(4):
        if args.phase == 'train':
            for j in range(20):
                if i == 0:
                    scenes.append("FloorPlan" + str(j + 1))
                else:
                    scenes.append("FloorPlan" + str(i + 1) + '%02d' % (j + 1))
        elif args.phase == 'eval':
            eval_scenes_list = []
            for j in range(10):
                if i == 0:
                    eval_scenes_list.append("FloorPlan" + str(j + 1 + 20))
                else:
                    eval_scenes_list.append("FloorPlan" + str(i + 1) + '%02d' % (j + 1 + 20))
            scenes.append(eval_scenes_list)

    return scenes


# from offline data to get depth map
def sarpn_depth_h5(state_name,scene_name):
    state_name = str(state_name)
    save_dir = '/home/chenhaobo996/hxb/Data/Data_ai2thor_depth_h5/'
    file_name = os.path.join(save_dir, scene_name, scene_name+'.h5')
    f = h5py.File(file_name, 'r')
    depth = f[state_name][:]     
    depth = torch.from_numpy(depth)
    return depth