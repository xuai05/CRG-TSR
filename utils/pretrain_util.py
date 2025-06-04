import os
import json

from yaml import FlowSequenceStartToken
import numpy as np

import torch
from torch.utils.data import Dataset

from datasets.constants import AI2THOR_TARGET_CLASSES

class PreModelTranfsDataset(Dataset):
    def __init__(self, args, data_type='train'):
        self.data_dir = args.data_dir
        self.targets_index = [i for i, item in enumerate(AI2THOR_TARGET_CLASSES[60]) if item in AI2THOR_TARGET_CLASSES[22]]

        self.annotation_file = os.path.join(self.data_dir, 'annotation_{}.json'.format(data_type))
        with open(self.annotation_file, 'r') as rf:
            self.annotations = json.load(rf)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        location = annotation['location']
        target = annotation['target']
        optimal_action = annotation['oa']
        location = location.replace("|","_")
        # annotation_path = os.path.join(self.data_dir, 'data', self.detection_alg, '{}.npz'.format(location))
        annotation_path = os.path.join(self.data_dir, 'Data', '{}.npz'.format(location))

        data = np.load(annotation_path)

        global_feature_list = data['resnet18_feature'][-1]

        #将detr feature转化为8维的节点图特征，return结果为8维的全局节点图，供之后模型中的图卷积使用
        detr_feature_list = torch.from_numpy(data['detr_feature']) 
        depth_feature_list = data["depth_feature"].squeeze(1).squeeze(1)

        # print("global_feature shape is {}".format(global_feature_list.shape))
        # print("detr_feature_list shape is {}".format(detr_feature_list.shape))

        # features = data['detr_feature'][:,:, :256]
        # scores = data['detr_feature'][:,:, 256]
        # labels = data['detr_feature'][:,:, 257]
        # bboxes = data['detr_feature'][:,:, 260:]
        # print(labels.shape)
        # print(labels)
        Traject_Graph_node_input = torch.zeros(8,22,261)

        for i in range(detr_feature_list.shape[0]):
            detr = detr_feature_list[i]
            depth = depth_feature_list[i]
            for j in range(detr_feature_list.shape[1]):          # 100*260
                # 提取有标签的点
                if detr[j][257] != 0 and detr[j][257]  != 23:           # 判断label存在
                    # 当前子图节点信息添加
                    graph_index = int(detr[j][257])-1
                    Traject_Graph_node_input[i][graph_index][0] = detr[j][257]              #label
                    Traject_Graph_node_input[i][graph_index][1] = detr[j][260]       # x_dim
                    Traject_Graph_node_input[i][graph_index][2] = detr[j][262] 
                    # Traject_Graph_node_input[i][graph_index][3] = get_depth(detr[j][260],detr[j][261],detr[j][262],detr[j][263],depth)        # depth
                    if detr[j][257] ==  (AI2THOR_TARGET_CLASSES[22].index(target) + 1):
                        Traject_Graph_node_input[i][graph_index][3] = get_depth(detr[j][260],detr[j][261],detr[j][262],detr[j][263],depth)        # depth       #仅给予目标节点以深度信息
                        Traject_Graph_node_input[i][graph_index][4] = 1                # 添加indicator信息，进行测试   
                    Traject_Graph_node_input[i][graph_index][5:261] = detr[j][:256]         # 特征项
        
        # # generate target indicator array based on detection results labels
        # target_embedding_array = np.zeros((8,data['detr_feature'].shape[1], 1))
        # target_embedding_array[labels[:] == (AI2THOR_TARGET_CLASSES[22].index(target) + 1)] = 1

    
        # local_feature = {
        #     'features': features,
        #     'scores': scores,
        #     'labels': labels,
        #     'bboxes': bboxes,
        #     'indicator': target_embedding_array,
        #     'locations': location,
        #     'targets': target,
        #     'idx': idx,
        # }

        return global_feature_list, Traject_Graph_node_input, optimal_action

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    # args.distributed = True
    args.distributed = False
    return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print



def get_depth(x1,y1,x2,y2,depth_feature):

    int_target0, int_target1, int_target2, int_target3  = int((x1*152)/300),int((y1*114)/300),int((x2*152)/300),int((y2*114)/300) 

    if int_target0<=0:int_target0=0
    if int_target1<=0:int_target1=0
    if int_target2<=0:int_target2=0
    if int_target3<=0:int_target3=0
    if int_target0 == int_target2:      #Prevent nan
        if int_target0 == 0:
            int_target2 = int_target2 + 1
        else:
            int_target0 = int_target0-1
    if int_target1 == int_target3: 
        if int_target1 == 0:
            int_target3 = int_target3 + 1
        else:
            int_target1 = int_target1-1
    depth = depth_feature[int_target1:int_target3,int_target0:int_target2].mean()

    if depth != depth:
        print("int_target0, int_target1, int_target2, int_target3 is {},{},{},{}".format(int_target0, int_target1, int_target2, int_target3))
        print("detection_inputs['depth'][i] is {}".format(depth))

    depth = torch.tensor(depth)
    return  depth

class PreVisTranfsDataset(Dataset):
    def __init__(self, args, data_type='train'):
        self.data_dir = args.data_dir

        self.targets_index = [i for i, item in enumerate(AI2THOR_TARGET_CLASSES[60]) if item in AI2THOR_TARGET_CLASSES[22]]

        self.annotation_file = os.path.join(self.data_dir, 'annotation_{}.json'.format(data_type))
        with open(self.annotation_file, 'r') as rf:
            self.annotations = json.load(rf)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        location = annotation['location']
        target = annotation['target']
        optimal_action = annotation['optimal_action']

        annotation_path = os.path.join(self.data_dir, 'data', '{}.npz'.format(location))

        data = np.load(annotation_path)

        global_feature = data['resnet18_feature']

        features = data['detr_feature'][:, :256]
        scores = data['detr_feature'][:, 256]
        labels = data['detr_feature'][:, 257]
        bboxes = data['detr_feature'][:, 260:]

        # generate target indicator array based on detection results labels
        target_embedding_array = np.zeros((data['detr_feature'].shape[0], 1))
        target_embedding_array[labels[:] == (AI2THOR_TARGET_CLASSES[22].index(target) + 1)] = 1

    

        local_feature = {
            'features': features,
            'scores': scores,
            'labels': labels,
            'bboxes': bboxes,
            'indicator': target_embedding_array,
            'locations': location,
            'targets': target,
            'idx': idx,
        }

        return global_feature, local_feature, optimal_action