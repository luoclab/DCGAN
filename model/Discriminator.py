from PIL import Image
import os

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets
from torch import nn
from data.loader_anime_face_dataset import AnimeDataset

class Discriminator(nn.Module):
    def __init__ (self,n_channels):
        super(Discriminator,self).__init__()
        #3*64*64->64*32*32
        self.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=64,
            kernel_size =4,
            stride =2,padding =1,bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.leakyrelu1= nn.LeakyReLU(0.2)
        
        #64*32*32->128*16*16
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size =4,
            stride =2,padding =1,bias = False)
        self.bn2 = nn.BatchNorm2d(128)
        self.leakyrelu2= nn.LeakyReLU(0.2)
        
        #128*16*16->256*4*4
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size =4,
            stride =2,padding =1,bias = False)
        self.bn3 = nn.BatchNorm2d(256)
        self.leakyrelu3= nn.LeakyReLU(0.2)
        
        #256*4*4->512*2*2
        self.conv4 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size =4,
            stride =2,padding =1,bias = False)
        self.bn4 = nn.BatchNorm2d(512)
        self.leakyrelu4= nn.LeakyReLU(0.2)
        
        #512*2*2->1024*1*1
        self.conv5 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size =4,
            stride =1,padding =0,bias = False)
        
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.leakyrelu1(x)
        
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.leakyrelu2(x)
        
        x=self.conv3(x)
        x=self.bn3(x)
        x=self.leakyrelu3(x)
        
        x=self.conv4(x)
        x=self.bn4(x)
        x=self.leakyrelu4(x)
        
        x=self.conv5(x)
        x=self.sigmoid(x)    
        return x.view(-1, 1).squeeze(1)


if __name__=='__main__':
    path=r'K:\Dataset\动漫头像\archive\images'
    trans=transforms.Compose([transforms.Resize((64,64)),
                              transforms.CenterCrop(64),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,0.5,0.5),
                                                        std=(.5,.5,.5))])
    dataset=AnimeDataset(path,trans)
    dl_train=DataLoader(dataset,batch_size=64,shuffle=True)
    features=next(iter(dl_train))
    print(features.shape)