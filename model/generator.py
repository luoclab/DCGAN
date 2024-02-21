from torch import nn

class Generator(nn.Module):
    def __init__(self,noise_size,n_channels):
        super(Generator,self).__init__()
        #将100个1*1的噪声向量z，转换为1024个4*4大小的特征图
        self.ct1=nn.ConvTranspose2d(#这里用了转置卷积，什么是转置卷积？
            in_channels=noise_size,
            out_channels=1024,
            kernel_size =4,
            stride =1,padding =0,bias = False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU()
        
        #将1024个4*4的特征图，转换为512个8*8的特征图
        self.ct2=nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size =4,
            stride =2,padding =1,bias = False)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU()
        
        #将512个8*8的特征图，转换为256个16*16的特征图
        self.ct3=nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size =4,
            stride =2,padding =1,bias = False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        
        #256个16*16的特征图，转换为128个32*32的特征图
        self.ct4=nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size =4,
            stride =2,padding =1,bias = False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        
        #128个32*32的特征图，转换为3个64*64的特征图
        self.ct5=nn.ConvTranspose2d(
            in_channels=128,
            out_channels=3,
            kernel_size =4,
            stride =2,padding =1,bias = False)
        
        self.Tanh=nn.Tanh()

    def forward(self,x):
        x=self.ct1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        
        x=self.ct2(x)
        x=self.bn2(x)
        x=self.relu2(x)

        x=self.ct3(x)
        x=self.bn3(x)
        x=self.relu3(x)

        x=self.ct4(x)
        x=self.bn4(x)
        x=self.relu4(x)

        x=self.ct5(x)
        x=self.Tanh(x)
        
        return x