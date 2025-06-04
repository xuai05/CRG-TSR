#此版本预训练4头2层Model和4头2层的transformer兵并且将Model的输出，进行信息补充，形成结构再喂给transformer

""" Borrowed from VTNet_model """
from __future__ import division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# from .Model_var import ASTGnetwork, get_gloabal_pos_embedding,gcn
from .Model import ASTGnetwork, get_gloabal_pos_embedding,gcn,VisualTransformer


class PreTrainedModel(nn.Module):
    def __init__(self, args):
        super(PreTrainedModel, self).__init__()
        self.image_size = 300

        # same layers as VisualTransformer visual representation learning part
        self.global_conv = nn.Conv2d(512, 256, 1)
        self.global_pos_embedding = get_gloabal_pos_embedding(7, 64)

        self.ASTGnetwork = ASTGnetwork(nb_head=4, num_layers=2)
        self.visual_transformer = VisualTransformer()
        # 自适应邻接矩阵可学习参数初始化
        self.num_nodes = 22
        self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes), requires_grad=True)

        # 图卷积层
        self.dilation_channels=261
        self.residual_channels=128
        self.dropout=0.3
        self.gconv = nn.ModuleList()
        self.gcn_layers_num = 2
        self.gcnet = gcn(self.dilation_channels,self.residual_channels,self.dropout)

        self.global_linear = nn.Linear(49, 22)
        self.global_conv = nn.Conv2d(512, 128, 1)
        # pretraining network action predictor, should be used in Visual Transformer model
        self.pretrain_fc = nn.Linear(6272, 6)

        self.graph_detection = nn.Sequential(
            nn.Linear(389, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )


    def forward(self, global_feature: torch.Tensor, local_feature: dict):   
        #global_feature[128, 1, 512, 7, 7]
        #local_feature shape is torch.Size([128, 8, 22, 261])
        batch_size = global_feature.shape[0]
        # print("global_feature shape is {}".format(global_feature.shape))
        # print("local_feature shape is {}".format(local_feature.shape))

        local_feature = local_feature[:,-5:-1,:,:]          # 4个轨迹步数
        
        # print("local_feature shape is {}".format(local_feature.shape))
        # global_feature = global_feature.squeeze(dim=2)
        # print("global_feature shape is {}".format(global_feature.shape))
        # image_embedding = F.relu(self.global_conv_fromresnet25(global_feature))

        # 获得邻接矩阵A
        self.nodevec1 = self.nodevec1
        self.nodevec2 = self.nodevec2
        Adj_matrix = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        #一层自适应的扩散图卷积获取当前子图表示
        Graph_node_input_ori = self.gcnet(local_feature, Adj_matrix)          # 22,128



        # 对全局特征做处理
        global_feature = global_feature.squeeze(dim=1)
        image_embedding = F.relu(self.global_conv(global_feature))       #get the gobal_feature  [128,512,7,7]->[128,128,7,7]
        image_embedding = image_embedding + self.global_pos_embedding.repeat([batch_size, 1, 1, 1]).cuda()          # [128, 128, 7, 7]
        image_embedding = image_embedding.reshape(batch_size,128, -1)        #[128, 128, 7, 7]->128*128, 7, 7]->128*128*49
        # image_embedding = F.relu(self.global_linear(image_embedding)).transpose(1,2).unsqueeze(dim=1)                    # 
        # image_embedding
        # ASTgraph_embedding = self.ASTGnetwork(Graph_node_input, image_embedding, Adj_matrix).squeeze(dim=1)                # ASTGnetwork input 128, 8, 22, 128]; 128, 1, 22, 128 output batch 22,1,128
        
        last_graph = Graph_node_input_ori[:,-1,:,:].unsqueeze(dim=1)
        # print("last_graph shape is {} Graph_node_input_ori shape is {}".format(last_graph.shape,Graph_node_input_ori.shape))
        ASTgraph_embedding = self.ASTGnetwork(Graph_node_input_ori,last_graph, Adj_matrix).squeeze(dim=2)         #                （） （128, 1, 22, 128） （22*22）->128*22*128
        

        #带有信息补充的接入到transformer中
        Graph_features = local_feature[:,-2,:,5:261].squeeze(dim=1)
        Graph_attention_input = torch.matmul(Graph_features.permute(0, 2, 1), Adj_matrix.squeeze(dim=0)).permute(0, 2, 1)

        Graph_attention_input = torch.cat((
            Graph_attention_input,
            ASTgraph_embedding.squeeze(dim=0),
            local_feature[:,-1,:,0:5].squeeze(dim=1)), dim=2)           # 128,22,356
        Graph_attention_input = F.relu(self.graph_detection(Graph_attention_input))         #  1*22*128

        # 用于学习当前帧内部的图像关系
        visual_representation = self.visual_transformer(src=Graph_attention_input,                   # 128, 22, 128
                                                                        query_embed=image_embedding)  # [128, 128, 49]

        visual_rep = visual_representation.reshape(batch_size, -1)
        action = self.pretrain_fc(visual_rep)

        return {
            'action': action,
            'fc_weights': self.pretrain_fc.weight,
            'visual_reps': visual_rep.reshape(batch_size, 49, 128)
        }





