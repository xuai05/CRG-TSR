from __future__ import division

import math
from operator import concat

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_util import norm_col_init, weights_init

from .model_io import ModelOutput
from .G import G, TransformerDecoderLayer, TransformerDecoder

from .TL import MultiHeadAttention, TemporalPositionalEncoding, SpatialPositionalEncoding, EncoderLayer, Encoder, \
    DecoderLayer, Decoder, EncoderDecoder, \
    PositionWiseGCNFeedForward, spatialAttentionScaledGCN
    # MultiHeadAttentionAwareTemporalContex_q1d_k1d, MultiHeadAttentionAwareTemporalContex_qc_k1d, MultiHeadAttentionAwareTemporalContex_qc_kc

import copy

# Graph_WaveNet
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        # print("x shape is {}".format(x.shape))            # 22,260
        # print("A shape is {}".format(A.shape))            # 22,22
        # x = torch.einsum('nf,nw->wf',(x,A))           # 无batch的计算
        x = torch.einsum('ncvl,vw->ncwl',(x,A))         # 有batch的计算,n batch;c 时间片
        return x.contiguous()

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order+1)*c_in
        self.mlp = nn.Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,apt):
        out = [x]
        x1 = self.nconv(x,apt)
        out.append(x1)
        for k in range(2, self.order + 1):          # 这一步相当于做了一个扩散卷积
            x2 = self.nconv(x1,apt)
            out.append(x2)
            x1 = x2
        h = torch.cat(out,dim=3)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class TLnetwork(nn.Module):
    def __init__(self, d_model=128, nb_head=4, num_layers=2, dropout=0.1,
                encoder_input_size = 128,decoder_output_size =128,num_of_vertices=22,num_for_embedding=1,
    ):
        super(TLnetwork, self).__init__()
        
        self.src_dense = nn.Linear(encoder_input_size, d_model)
        self.trg_dense = nn.Linear(decoder_output_size, d_model)  # target input projection
        c = copy.deepcopy
        self.position_wise_gcn = PositionWiseGCNFeedForward(spatialAttentionScaledGCN(d_model, d_model), dropout=dropout)
        
        # 先尝试使用普通的注意力机制
        self.attn_ss = MultiHeadAttention(nb_head, d_model, dropout=dropout)
        self.attn_st = MultiHeadAttention(nb_head, d_model, dropout=dropout)
        self.att_tt = MultiHeadAttention(nb_head, d_model, dropout=dropout)
        
        # # 时间位置编码————————是否需要？？？？
        self.encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, 8)  # decoder temporal position embedding
        # self.decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_embedding)

        # # 空间位置编码————————编码到节点的维度上
        self.spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout, self.position_wise_gcn, smooth_layer_num=0)

        # 节点嵌入
        self.encoder_embedding = nn.Sequential(self.src_dense, c(self.encode_temporal_position), c(self.spatial_position))
        self.decoder_embedding = nn.Sequential(self.trg_dense, c(self.spatial_position))

        # 编码器——————对于编码器中的卷积层，先使用
        self.encoderLayer = EncoderLayer(d_model, self.attn_ss, self.position_wise_gcn, dropout, residual_connection=True, use_LayerNorm=True)

        self.encoder = Encoder(self.encoderLayer, num_layers)
        # # 解码器
        self.decoderLayer = DecoderLayer(d_model, self.att_tt, self.attn_st, self.position_wise_gcn, dropout, residual_connection=True, use_LayerNorm=True)

        self.decoder = Decoder(self.decoderLayer, num_layers)

        self.generator = nn.Linear(d_model, decoder_output_size)


        self.model = EncoderDecoder(self.encoder,
                            self.decoder,
                            self.encoder_embedding,
                            self.decoder_embedding,
                            self.generator)
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, src, trg, Adj_matrix):
        # out (nodes_num, time_num, d_models)       22,8,128
        # Adj_matrix (nodes_num,nodes_num)             22*22

        src = src.transpose(1,2)
        trg = trg.transpose(1,2)
        # src = self.encoder(out,Adj_matrix)
        # print("encoder out shape is {}".format(out.shape))
        # out = self.decoder(out,Adj_matrix)
        out = self.model(src, trg, Adj_matrix)
        # self.Adj_matrix = Adj_matrix,
        # print("out of ModelNN is {}".format(out))
        return out

class GlobalAttention(G):
    def __init__(self, d_model=128, nhead=4, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=128, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super(GlobalAttention, self).__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, query_embed):
        bs, n, c = src.shape
        src = src.permute(1, 0, 2)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, src, query_pos=query_embed)
        return hs.transpose(0, 1)



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        # 自适应邻接矩阵可学习参数初始化
        self.num_nodes = 22
        self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes), requires_grad=True)

        # 图卷积层
        self.dilation_channels=263
        self.residual_channels=128
        self.dropout=0.3
        self.gconv = nn.ModuleList()
        self.gcn_layers_num = 2
        self.gcnet = gcn(self.dilation_channels,self.residual_channels,self.dropout)


        # 时空注意力图卷积网络
        self.TLnetwork = TLnetwork()
        self.detection_map = nn.Linear(100, 22)
        self.node_linear = nn.Linear(256,128)


        self.Global_Attention = GlobalAttention()

        self.num_cate = args.num_category
        self.image_size = 300

        # global visual representation learning networks
        resnet_embedding_sz = 512
        hidden_state_sz = args.hidden_state_sz
        self.global_conv = nn.Conv2d(resnet_embedding_sz, 128, 1)
        self.global_linear = nn.Linear(49, 22)
        self.global_pos_embedding = get_gloabal_pos_embedding(7, 64)


        self.graph_detection = nn.Sequential(
            nn.Linear(391, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # previous action embedding networks
        action_space = args.action_space
        self.embed_action = nn.Linear(action_space, 128)


        # ==================================================
        # navigation policy part
        self.lstm_input_sz = 6400
        self.hidden_state_sz = hidden_state_sz

        self.lstm = nn.LSTM(self.lstm_input_sz, hidden_state_sz, 2)

        self.critic_linear_1 = nn.Linear(hidden_state_sz, 64)
        self.critic_linear_2 = nn.Linear(64, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, action_space)


        # ==================================================
        # weights initialization
        # self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.global_conv.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear_1.weight.data = norm_col_init(
            self.critic_linear_1.weight.data, 1.0
        )
        self.critic_linear_1.bias.data.fill_(0)
        self.critic_linear_2.weight.data = norm_col_init(
            self.critic_linear_2.weight.data, 1.0
        )
        self.critic_linear_2.bias.data.fill_(0)

        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_ih_l1.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.lstm.bias_hh_l1.data.fill_(0)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def embedding(self, state, detection_inputs,Temporal_memory, action_embedding_input):
        # detection_inputs contains the features embedding [100, 256], the scores [100], the labels [100], teh bboxes [100,4] is the Center coordinates and height width,the indicator [100,1]
        # what is the target [9],tensor([-2.4938e+35,  4.5818e-41, -1.6821e-19,  3.0966e-41,  1.4013e-45, 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00], device='cuda:0')????


        action_embedding = F.relu(self.embed_action(action_embedding_input))            # action_embedding shape is torch.Size([1, 128])
        # 初始化当前输入子图
        gpu_id = state.get_device()
        Graph_node_input_ori = torch.zeros(22,263).cuda(gpu_id)
        for i in range(len(detection_inputs['indicator'])):
            # 提取有标签的点
            if detection_inputs['labels'][i] != 0 and detection_inputs['labels'][i] != 23:
                # 当前子图节点信息添加_
                graph_index = int(detection_inputs['labels'][i])-1
                Graph_node_input_ori[graph_index][0] = detection_inputs['labels'][i]
                Graph_node_input_ori[graph_index][1:3] = detection_inputs["x_dim"][i]
                Graph_node_input_ori[graph_index][3:5] = detection_inputs["y_dim"][i]           # 新增y方向信息
                Graph_node_input_ori[graph_index][5] = detection_inputs["depth"][i]
                Graph_node_input_ori[graph_index][6] = detection_inputs['indicator'][i]               
                Graph_node_input_ori[graph_index][7:263] = detection_inputs['features'][i]
        
        # print("Graph_node_input is {}".format(Graph_node_input))
        # 获得邻接矩阵A
        self.nodevec1 = self.nodevec1.cuda(gpu_id)
        self.nodevec2 = self.nodevec2.cuda(gpu_id)
        Adj_matrix = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        #一层自适应的扩散图卷积获取当前子图表示
        Graph_node_input = Graph_node_input_ori.unsqueeze(dim=0).unsqueeze(dim=0)       # 1,1,22,128
        Graph_node_input = self.gcnet(Graph_node_input, Adj_matrix).squeeze(dim=0)          # 1,1,22,128

        # 添加action信息
        append_action = action_embedding.repeat(22,1).unsqueeze(dim=0)
        Graph_node_input  = self.node_linear(torch.cat((Graph_node_input,append_action),dim=2))

        # 构造全局图时间序列
        src = torch.cat((Temporal_memory,Graph_node_input),dim=0)[1:].unsqueeze(dim=0)      # batch,time,22,128(1,8,22,128)
        # 对全局特征做处理
        image_embedding = F.relu(self.global_conv(state))       #get the gobal_feature  [1,512,7,7]->[1,128,7,7]
        image_embedding = image_embedding + self.global_pos_embedding.cuda(gpu_id)          # [1, 128, 7, 7]
        image_embedding = image_embedding.reshape(1,128, -1)        #[1, 128, 7, 7]->128, 7, 7]->128*49
        # image_embedding = F.relu(self.global_linear(image_embedding))                 

        # 时空注意力transformer->对当前帧的图关系作为decoder的输入，放入到ASTGCN中,用于学习不同帧图之间的关系
        ASTgraph_embedding = self.TLnetwork(src, Graph_node_input.unsqueeze(dim=0), Adj_matrix).squeeze(dim=2)         # 

        #带有信息补充的接入到transformer中
        Graph_features = Graph_node_input_ori[:,7:263]
        Graph_attention_input = torch.mm(Graph_features.t(), Adj_matrix.squeeze(dim=0)).t()   
        Graph_attention_input = torch.cat((
            Graph_attention_input,
            ASTgraph_embedding.squeeze(dim=0),
            Graph_node_input_ori[:,0:7]), dim=1).unsqueeze(dim=0)           # 1,22,391
        Graph_attention_input = F.relu(self.graph_detection(Graph_attention_input))         #  1*22*128
        
        # 用于学习当前帧内部的图像关系
        visual_representation = self.Global_Attention(src=Graph_attention_input,                  # 1, 22, 128
                                                                        query_embed=image_embedding)            # 1, 128, 49

        out = torch.cat((visual_representation.squeeze(dim=0), action_embedding), dim=0)             
        
        out = out.reshape(1, -1)        # [1, 6400]
        return out, image_embedding,Temporal_memory

    def a3clstm(self, embedding, prev_hidden_h, prev_hidden_c):         # 此部分的lstm是考虑了历史的动作和状态信息，和本文的时空图并不冲突，暂时不删除
        embedding = embedding.reshape([1, 1, self.lstm_input_sz])
        output, (hx, cx) = self.lstm(embedding, (prev_hidden_h, prev_hidden_c))
        x = output.reshape([1, self.hidden_state_sz])           # x shape [1, 512]

        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear_1(x)
        critic_out = self.critic_linear_2(critic_out)

        return actor_out, critic_out, (hx, cx)





    def attention_map(self, query, key, value): # 49,256;    22,256;   22,256;  
        d_k = query.size(-1)        # 256
        scores = torch.matmul(query, key.transpose(0,1) ) / math.sqrt(d_k)  
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value)

    def forward(self, model_input, model_options):
        state = model_input.state
        # state is torch.Size([1, 512, 7, 7])
        (hx, cx) = model_input.hidden
        Temporal_memory = model_input.graphmemory 
        detection_inputs = model_input.detection_inputs
        action_probs = model_input.action_probs
        x, image_embedding,Temporal_memory = self.embedding(state, detection_inputs, Temporal_memory, action_probs)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, hx, cx)

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            graphmemory=Temporal_memory,
            embedding=image_embedding,
        )

#position embedding
def get_gloabal_pos_embedding(size_feature_map, c_pos_embedding):
    mask = torch.ones(1, size_feature_map, size_feature_map)

    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)

    dim_t = torch.arange(c_pos_embedding, dtype=torch.float32)

    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / c_pos_embedding)           #torch==1.9
    # dim_t = 10000 ** (2 * (dim_t // 2) / c_pos_embedding)           # torch == 1.4

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos

