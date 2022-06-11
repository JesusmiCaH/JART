import torch
import torch.nn as nn
import numpy as np
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from model import ViT
from UCF import UCF101,ClipSubstractMean,Rescale,RandomCrop,ToTensor
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import gc

import wandb

# # set hyperparameters
# sweep_config={
#     'method':'random',
#     'metric':{
#         #metric：目标是最大化acc。
#         'name':'val-acc',
#         'goal':'maximize',
#     },
# }
# hyper_config={
#     'learning_rate':{
#         #学习率：从0到10指数取值。
#         'distribution':'log_uniform_values',
#         'min':1e-4,
#         'max':3e-3,
#     },
#     'dropout_rate':{
#         #dropout_rate：0到0.5，均匀分布。
#         'distribution':'uniform',
#         'min':0.1,
#         'max':0.5,
#     },
#     'THW_clip':{
#         #视频切割方法，列表中三个值乘出来就是序列长度。
#         'values':[(4,10,10),(8,8,8),(16,5,5)]
#     },
#     'd_model':{
#         #d_model维度,从64到512。
#         'distribution':'q_log_uniform_values',
#         'q':8,
#         'min':32,
#         'max':256,
#     },
#     'encoder_mlp_dim':{
#         #encoder中MLP层的中间维度，要注意MLP层的结构是两层FC夹GELU
#         'values':[512,1024,2048]
#     },
#     'ViT_nhead':{
#         #nhead头数：2，4，8，16。
#         'values':[2,4,8]
#     },
#     'num_encoder_layers':{
#         #encoderlayer层数：1到4。
#         'values':[1,2,3,4]
#     },
#     'train_test_list':{
#         #用哪组train/test列表。
#         'values':['01', '02', '03']
#     },
# }
# sweep_config['parameters']=hyper_config
# sweep_config['name']='short_term_ver'
# sweeper = wandb.sweep(sweep_config, project='my-vit')

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 阶梯式下降学习率函数
def get_stair_schedule(optimizer, drop_epoches = [40,80], epoch_step=100, drop_rate = 10.):
    def lr_lambda(current_step):
        for idx, drop_epoch in enumerate(drop_epoches):
            if current_step >= drop_epoch*epoch_step:
                pass
            else:
                return (1/drop_rate)**idx
        return (1/drop_rate)**len(drop_epoches)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda,-1)

def train():
    torch.cuda.empty_cache()
    gc.collect()
    with wandb.init(
        project = 'my-vit',
        name = 'formal_experiment',
        config = {
            'learning_rate':0.00023,
            'dropout_rate':0.103,
            'THW_clip':(8,8,8),
            'd_model':104,
            'encoder_mlp_dim':512,
            'ViT_nhead':8,
            'num_encoder_layers':3,
            'train_test_list':'01',
        }
    ):
        config = wandb.config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {device} now!')

        # 加载数据集
        train_list_path = './ucfTrainTestlist/trainlist'+config['train_test_list']+'.txt'
        # test_list_path = './ucfTrainTestlist/testlist'+config['train_test_list']+'.txt'
        root_path = '/home/featurize/data/UCF-101-jpg/'
        myUCF101 = UCF101(train_list_path, root_path, transform=torchvision.transforms.Compose([ClipSubstractMean(),Rescale(),RandomCrop(),ToTensor()]))
        # test_UCF101 = UCF101(test_list_path, root_path, transform=torchvision.transforms.Compose([ClipSubstractMean(),Rescale(),RandomCrop(),ToTensor()]))
        
        train_len = int(0.9*len(myUCF101))
        train_UCF101, val_UCF101 = random_split(myUCF101, [train_len, len(myUCF101)-train_len])
        
        batchsize=64
        epoch_step = len(train_UCF101)//batchsize + 1  #每一轮epoch需要跑多少个batch。

        train_loader = DataLoader(train_UCF101, batch_size=batchsize,shuffle=True)
        val_loader = DataLoader(val_UCF101, batch_size=batchsize,shuffle=False)

        
        # 模型定义
        model = ViT([16,160,160], config['THW_clip'], config['d_model'], 10000, config['ViT_nhead'], config['encoder_mlp_dim'], config['num_encoder_layers'], 101, config['dropout_rate'], device).to(device)
        # 加载预训练权重
        model.load_state_dict(torch.load('./models/4e-05_Epoch5.pth'))
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        # 学习率调整策略
        # scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=10, num_training_steps=100, num_cycles=0.5)
        # scheduler = get_stair_schedule(optimizer, [40,80], epoch_step, 10.)
        scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=epoch_step)
        
        wandb.watch(model,log_freq=100)
        best_acc = 0.
        for epoch in range(100):
            train_loss = 0
            train_acc = 0
            val_loss = 0
            val_acc = 0
            # training
            model.train()
            for (idx, batch) in enumerate(train_loader):
                optimizer.zero_grad()
                
                video_x = batch['video_x'].to(device)
                # print(video_x)
                video_label = batch['video_label'].to(device)

                video_y = model(video_x)

                # get Loss and Acc
                loss = criterion(video_y, video_label)
                # print(torch.argmax(video_y, dim=1)==video_label)
                acc = torch.sum( torch.argmax(video_y, dim=1)==video_label )
                
                print("Batch {2}/{3}:Loss is {0}, acc is {1}, lr is {4}.".format(loss.item()/batchsize, acc/batchsize, idx, len(train_loader), scheduler.get_lr()))
                train_loss+=loss.item()
                train_acc+=acc
                
                # optimize
                loss.backward()
                optimizer.step()
                scheduler.step()
            round_train_acc = train_acc/len(train_UCF101)
            print('------------------------------------------------------')
            print("Train acc is {0}".format(round_train_acc))
            print('------------------------------------------------------')
            # validation
            if len(val_UCF101)>0:
                model.eval()
                for (idx, batch) in enumerate(val_loader):
                    video_x = batch['video_x'].to(device)
                    video_label = batch['video_label'].to(device)

                    video_y = model(video_x)

                    # get Loss and Acc
                    loss = criterion(video_y, video_label)
                    acc = torch.sum( torch.argmax(video_y, dim=1)==video_label )
                    
                    print("Batch {2}/{3}:Loss is {0}, acc is {1}.".format(loss.item()/batchsize, acc/batchsize, idx, len(val_loader)))
                    val_loss+=loss.item()
                    val_acc+=acc
            
            round_val_acc = val_acc/len(val_UCF101)
            # 注意这里已经走出了batch小循环，要记录下训练和验证的成绩。
            print('------------------------------------------------------')
            print("Test acc is {0}".format(round_val_acc))
            print('------------------------------------------------------')
            wandb.log({
                'train-loss':train_loss,
                'train-acc':round_train_acc,
                'val-loss':val_loss,
                'val-acc':round_val_acc,
                })
            if round_val_acc>0.2 and round_val_acc > best_acc:
                best_acc = round_val_acc
                # 权重文件保存路径
                model_path = './models/{0}_Epoch{1}.pth'.format(config['learning_rate'],epoch)
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc))
                

if __name__ == '__main__':
    train()
