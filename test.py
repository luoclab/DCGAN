import torch
from torchvision import utils
from model.generator import Generator
import os
if __name__=='__main__':
    #噪声向量维度
    noise_size=100
    n_channels=3#输入通道数
    #将它们传入Generator，定义netG
    # netG=Generator(noise_size,n_channels)
    # netG.load_state_dict(torch.load('weight\model_weights.pth'))
    #调整为评估模式# 定义随机噪声
    model_path = os.path.join('weight', "modelsG.pt")
    netG = torch.load(model_path)

    # 将模型设置为评估模式（如果只是推断，而不是训练）
    netG.eval()
    # 读取已经训练的生成模型
    fixed_noise =torch.randn(16,noise_size,1,1)
    fake =netG(fixed_noise)#传入netG生成假图fake# 调用save image保存图像
    utils.save_image(fake.detach(),'anime-face.png', normalize=True)