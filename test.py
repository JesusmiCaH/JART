import torch
import torch.nn as nn
import numpy as np
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from model import ViT
from UCF import UCF101,ClipSubstractMean,Rescale,RandomCrop,ToTensor
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

import wandb

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def test():

    config = {
            'learning_rate':0.000008,
            'dropout_rate':0.103,
            'THW_clip':(8,8,8),
            'd_model':104,
            'encoder_mlp_dim':512,
            'ViT_nhead':8,
            'num_encoder_layers':3,
            'train_test_list':'01',
        }
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

    batchsize=32
    epoch_step = len(train_UCF101)//batchsize + 1  #每一轮epoch需要跑多少个batch。

    val_loader = DataLoader(val_UCF101, batch_size=batchsize,shuffle=False)

    # 模型定义
    model = ViT([16,160,160], config['THW_clip'], config['d_model'], 10000, config['ViT_nhead'], config['encoder_mlp_dim'], config['num_encoder_layers'], 101, config['dropout_rate'], device).to(device)
    # 加载预训练权重
    model.load_state_dict(torch.load('./models/4e-05_Epoch5.pth'))
    
    confusion_matrix = torch.zeros(101,101)
    
    for epoch in range(1):
        val_acc_1 = 0
        val_acc_5 = 0
        # validation
        model.eval()
        for (idx, batch) in enumerate(val_loader):
            video_x = batch['video_x'].to(device)
            video_label = batch['video_label'].to(device)
            video_y = model(video_x)

            # get top1 Acc
            pred_label = torch.argmax(video_y, dim=1)
            acc_1 = torch.sum( pred_label==video_label )
            # get top5 Acc
            _, cls_5 = video_y.topk(5,1)
            row_label = video_label.view(-1,1)
            acc_5 = torch.eq(cls_5, row_label).sum()
            
            # upload confusion matrix
            confusion_matrix[video_label, pred_label]+=1
                
            print("Batch {2}/{3}:Top1 acc is {0}, Top5 acc is {1}.".format(acc_1/batchsize,acc_5/batchsize, idx, len(val_loader)))
            val_acc_1+=acc_1
            val_acc_5+=acc_5

        val_acc_1 = val_acc_1/len(val_UCF101)
        val_acc_5 = val_acc_5/len(val_UCF101)
        # 注意这里已经走出了batch小循环，要记录下训练和验证的成绩。
        print('------------------------------------------------------')
        print("Top1 acc is {0}, Top5 acc is {1}".format(val_acc_1, val_acc_5))
        print('------------------------------------------------------')
        
        confusion_matrix = confusion_matrix.numpy()
        num_per_cls=confusion_matrix.sum(1).reshape(-1,1)
        confusion_matrix /= num_per_cls
        plt.imshow(confusion_matrix)
        plt.savefig('./confuse.jpg')
        np.save('a_beautiful_line.npy',confusion_matrix)


if __name__ == '__main__':
    test()
