from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode



class ClipSubstractMean(object):
    def __init__(self, b=104, g=117, r=123):
        self.means = np.array((r, g, b))

    def __call__(self, video_x):
        return video_x - self.means


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=(182,242)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, video_x):
    
        h, w = video_x.shape[1], video_x.shape[2]
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        new_video_x=np.zeros((16,new_h,new_w,3))
        for i in range(16):
            image=video_x[i,:,:,:]
            img = transform.resize(image, (new_h, new_w))
            new_video_x[i,:,:,:]=img

        return new_video_x


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(160,160)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, video_x):

        h, w = video_x.shape[1],video_x.shape[2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
         
        new_video_x=np.zeros((16,new_h,new_w,3))
        for i in range(16):
            image=video_x[i,:,:,:]
            image = image[top: top + new_h,left: left + new_w]
            new_video_x[i,:,:,:]=image

        return new_video_x


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, video_x):

        # swap color axis because
        # numpy image: batch_size x H x W x C
        # torch image: batch_size x C X H X W
        video_x = video_x.transpose((0, 3, 1, 2))
        video_x = np.array(video_x)

        return torch.tensor(video_x)

# class Video2Seq(object):
#     def __init__(self, THW_clip):
#         self.THW_clip=THW_clip
    
#     def __call__(self, input):
#         # input: T*H*W*C
#         T, C, H, W = input.shape
#         T_clip, H_clip, W_clip = self.THW_clip
#         T_patch, H_patch, W_patch = T//T_clip, H//H_clip, W//W_clip
        
#         src_len = T_clip*H_clip*W_clip
#         output_seq = torch.zeros((src_len, T_patch, C, H_patch, W_patch))

#         for i in range(T_clip):
#             for j in range(H_clip):
#                 for k in range(W_clip):
#                     idx = k + j*W_clip + i*W_clip*H_clip
#                     output_seq[idx, :, :, :, :] = input[i*T_patch:(i+1)*T_patch, :, j*H_patch:(j+1)*H_patch, k*W_patch:(k+1)*W_patch]

#         return output_seq

    
class UCF101(Dataset):
    """UCF101 Landmarks dataset."""

    def __init__(self, info_path, root_dir, transform=None):
        """
        Args:
            info_list (string): Path to the info list file with annotations.
            root_dir (string): Directory with all the video frames.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_list = self.read_txt_list(info_path)

        self.root_dir = root_dir
        self.transform = transform
            
    def __len__(self):
        return len(self.input_list[0])

    # get (16,240,320,3)
    def __getitem__(self, idx):
        video_path = self.input_list[0][idx]

        video_label=self.input_list[1][idx]
        video_x=self.get_single_video_x(video_path)
        if self.transform:
            video_x = self.transform(video_x)

        sample = {'video_x':video_x, 'video_label':torch.tensor(video_label)}
            
        return sample


    def get_single_video_x(self,video_path):
        video_path = os.path.join(self.root_dir, video_path)
        # get the random 16 frame
        frame_count=len(os.listdir(video_path))
        
        image_path=os.path.join(video_path,'image_00001.jpg')
        test_img = io.imread(image_path)
        H,W,_ = test_img.shape
        video_x=np.zeros((16,H,W,3))
        
        max_step = frame_count//16
        sample_step = random.randint(1,max_step)

        image_start=random.randint(0,frame_count-16*sample_step)+1
        image_id=image_start

        for i in range(16):
            s="%05d" % image_id
            image_name='image_'+s+'.jpg'
            image_path=os.path.join(video_path,image_name)
            tmp_image = io.imread(image_path)
            video_x[i,:,:,:]=tmp_image
            image_id+=sample_step
        return video_x
    
    def read_txt_list(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            input_list = [[],[]]  #第一个放文件名，第二个放label。
            for line in lines:
                filename, label = line.split(' ')
                input_list[0].append(filename[:-4])  #切掉avi后缀，因为我们用的是文件夹。
                input_list[1].append(int(label)-1)  #label信息得改为int型，同时需要减一，因为torch分类是从0开始的。
        return input_list


if __name__=='__main__':
    #usage
    root_list='/home/featurize/data/UCF-101-jpg/'
    info_list='./ucfTrainTestlist/trainlist01.txt'
    myUCF101=UCF101(info_list,root_list,transform=transforms.Compose([ClipSubstractMean(),Rescale(),RandomCrop(),ToTensor()]))

    dataloader=DataLoader(myUCF101,batch_size=8,shuffle=True,num_workers=8)
    for i_batch,sample_batched in enumerate(dataloader):
        print(i_batch,sample_batched['video_x'].size(),sample_batched['video_label'].size())
  