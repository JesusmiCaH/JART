from re import L
from cv2 import exp
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def trig_positional_embedding(src_len, d_model):
    # 注意这里序列长度src_len的大小不能超过10000*2*pi，因为下面设定的数字是10000。
    embedding = np.zeros((src_len, d_model))
    # 偶数sin，奇数cos！注意d_model不可能是奇数！
    for k in range(d_model//2):
        omiga = np.power(10000, 2.*k/d_model)
        embedding[: , 2*k] = np.sin(np.arange(src_len)/omiga)
        embedding[: , 2*k+1] = np.cos(np.arange(src_len)/omiga)
    return torch.tensor(embedding, dtype=torch.float32)


class ViT(nn.Module):
    '''
    Step1: flatten and thus map the input patch into d-dim.
    Step2: attach a randint vector[cls] before the 1st vector of the sequence.
    Step2.5: POSITIONAL ENBEDDING
    Step3: send the whole sequence into encoder.
    Step4: take the 0th vector[cls] of output sequence and give it a LayerNorm.
    Step5: send it into MLP-classification-head.
    
    THW: a tuple describes the way to clip raw video, the num represent the num of patches in a dim.
    d_model: the dimension of the vector passed in the model.
    n_head: num of multi-head.
    encoder_mlp_dim: dim of the mlp layer in the encoder.
    num_layers: the number of layers.
    num_classes: the number of classes.
    dropout_rate: rate of 2 dropout layer.
    '''
    def __init__(self, input_THW, THW_clip, d_model, max_src_len, nhead, encoder_mlp_dim, num_layers, num_classes, dropout_rate, device):
        super().__init__()
        self.flatten = nn.Flatten(2)  # 从第二维开始flatten，因为前两维是batchsize和seqlen，动不得！

        # patch_size = input_THW[0]//THW_clip[0] * input_THW[1]//THW_clip[1] * input_THW[2]//THW_clip[2] * 3  
        patch_size = torch.prod(torch.tensor([input_THW[i]//THW_clip[i] for i in range(len(THW_clip))])) * 3   #这个3是channel数量。
        self.THW_clip = THW_clip
        self.d_model = d_model
        self.input_mapping = nn.Linear(patch_size,d_model)
        self.cls_token = nn.Parameter(torch.randn(1,1,d_model))
        self.positional_embedding=trig_positional_embedding(max_src_len+1, self.d_model).to(device)

        self.encoder = nn.Sequential(*[nn.TransformerEncoderLayer(d_model, nhead, encoder_mlp_dim) for _ in range(num_layers)])
        
        self.last_LN = nn.LayerNorm(d_model)
        # 这个头在train和finetune的时候是不一样的，可能会更换。
        self.cls_head = nn.Linear(d_model, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)  # 预定在input mapping前加入dropout。
        
        self.device = device

    def forward(self, x):
        patch_seq = self.Video2Seq(x, self.THW_clip)
        # 输入的patch_seq是六维的，分别为：batch_size*seq_len*frame_num*C*H*W
        batch_size, src_len, _,_,_,_ = patch_seq.shape
        seq = self.flatten(patch_seq)
        seq = self.dropout(seq)  # 加一点dropout防止过拟合。
        
        encoder_seq = self.input_mapping(seq)
        # encoder_seq的尺寸是 batch_size*seq_len*d_model
        encoder_seq = torch.concat( (self.cls_token.repeat((batch_size,1,1)),encoder_seq) , dim=1)
        # 添加positional enbedding
        # positional_embedding = trig_positional_embedding(src_len+1, self.d_model)
        
        encoder_seq = self.positional_embedding[:src_len+1, :] + encoder_seq
        
        encoder_seq = encoder_seq.transpose(0,1)
        output_seq = self.encoder(encoder_seq)
        output_seq = output_seq.transpose(0,1)
        
        y = output_seq[:,0,:].squeeze(1)
        y = self.last_LN(y)

        y_cls = self.cls_head(y)

        # return self.softmax(y_cls)
        return y_cls

    def Video2Seq(self, input, THW_clip):
        # input: T*C*H*W
        # 具体是怎么个排布，记得等会看一下！
        batchsize, T, C, H, W = input.shape
        T_clip, H_clip, W_clip = THW_clip
        T_patch, H_patch, W_patch = T//T_clip, H//H_clip, W//W_clip
        
        src_len = T_clip*H_clip*W_clip
        output_seq = torch.zeros((batchsize, src_len, T_patch, C, H_patch, W_patch), device = self.device)

        for i in range(T_clip):
            for j in range(H_clip):
                for k in range(W_clip):
                    idx = k + j*W_clip + i*W_clip*H_clip
                    output_seq[:, idx, :, :, :, :] = input[:, i*T_patch:(i+1)*T_patch, :, j*H_patch:(j+1)*H_patch, k*W_patch:(k+1)*W_patch]

        return output_seq


class I3D(nn.Module):
    
    def __init__(self, channel_list, num_classes, dropout_rate):
        super().__init__()
        VGG_blocks = []
        for i in range(len(channel_list)-1):
            in_channel, out_channel = channel_list[i:i+2]
            block = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, 3, 1, 1),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
                nn.MaxPool3d((2,2,1))
            )
            VGG_blocks.append(block)
        self.VGG = nn.Sequential(*VGG_blocks)

        self.dl = nn.Sequential(
            nn.Linear( int(channel_list[-1]* (160*0.5**(len(channel_list)-1))**2 * 16) , 1024),  #
            nn.ReLU(),
            nn.Linear(1024 , 512),
            nn.ReLU(),
            nn.Linear(512,num_classes)
        )

    def forward(self, x):
        # 注意输入的轨道顺序是：batch, frames, channels, h, w.
        x = x.permute(0,2,3,4,1)
        y = self.VGG(x)
        
        y = nn.Flatten(1)(y)
        print(y.shape)
        y = self.dl(y)
        return y


if __name__ == '__main__':

    test_input = torch.rand((32,16,3,160,160))
    patch_size=torch.prod(torch.tensor(test_input.shape[2:]))
    model = ViT((16,160,160),(4,10,10), 80, 1000, 2, 2048, 3, 10, 0.1, torch.device('cpu'))
    # model = I3D([3,64,128,256,512,512], 10, 0.1)

    y = model(test_input)
    print(next(model.parameters()).device)