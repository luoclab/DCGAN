from PIL import Image
import os


from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets
from torch import nn
import torch
from data.loader_anime_face_dataset import AnimeDataset
from torch import utils
from model.generator import Generator
from model.Discriminator import Discriminator
import torch.optim as optim
import torchvision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# custom(自定义) initial weight,called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__  # return m's name
    if classname.find('Conv') != -1:  # find():find 'classname' whether contains "Conv" character,if not,return -1;otherwise,return 0
        torch.nn.init.normal_(m.weight, 0.0, 0.02)  # nn.init.normal_():the initialization weigts used Normally Distributed,mean=0.0,std=0.02
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
        
fixed_noise = torch.randn(64, 100, 1, 1, device=device)


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
    
    noise_size=100
    n_channels=3#表示输入数据的通道数，通道数为3即为RGB彩色数据#定义生成器netG和判别器netD
    netG=Generator(noise_size,n_channels).to(device)
    # netG.apply(weights_init)#使用apply对网络初始化
    netG.load_state_dict(torch.load('weight\model_weights.pth'))
    netD= Discriminator(n_channels).to(device)
    netD.apply(weights_init)
    
    optimizerD=optim.Adam(netD.parameters(),
                          lr=0.0002,
                          betas=(0.5,0.999))
    
    optimizerG=optim.Adam(netG.parameters(),
                          lr=0.0002,
                          betas=(0.5,0.999))
    criterion=nn.BCELoss()
    
    n_epoch=1#
    for epoch in range(n_epoch):
        for i,data in enumerate(dl_train):
            data= data.to(device)
            optimizerD.zero_grad()#清空判别器中的梯度
            output=netD(data)#判断真实样本数据
            real_loss = criterion(output, torch.ones_like(output))##计算output和标签1之间的损失，基于真实数据，优化判别器netp的效果
            noise1 = torch.randn(64, noise_size, 1, 1, device=device)
            fake1 = netG(noise1)#输入至生成器，生成假图
            output =netD(fake1)#
            fake_loss = criterion(output,torch.zeros_like(output))
            loss_D=real_loss +fake_loss
            loss_D.backward()#计算模型参数的梯度
            optimizerD.step()
            
            optimizerG.zero_grad()
            noise2 = torch.randn(64,noise_size,1,1,device=device)
            fake2 = netG(noise2)#输入至生成器，生成假图
            output = netD(fake2)
            loss_G=criterion(output,torch.ones_like(output))
            loss_G.backward()#计算模型参数的梯度
            optimizerG.step()
            
            if i % 10 == 0:
                out_dir='image'
                print('111111', (epoch, n_epoch, i, len(dl_train), loss_D.item(), loss_G.item()))
                torchvision.utils.save_image(data, '%s/real.png'% out_dir, normalize=True)
                fake = netG(fixed_noise)
                torchvision.utils.save_image(fake.detach(), '%s/fake_epoch_%d.png'% (out_dir, epoch), normalize=True)
    torch.save(netG.cpu(), os.path.join('weight/model', "modelsG.pt"))
    torch.save(netD.cpu(), os.path.join('weight/model', "modelsD.pt"))
    torch.save(netG.cpu().state_dict(), os.path.join('weight/model', "models_dictG.pt"))
    torch.save(netD.cpu().state_dict(), os.path.join('weight/model', "models_dictD.pt"))

