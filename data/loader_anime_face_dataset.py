from PIL import Image
import os

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets

class AnimeDataset(Dataset):
    def __init__(self,data_dir,transform):
        self.file_list=list()
        files=os.listdir(data_dir)
        for file in files:
            path=os.path.join(data_dir,file)
            self.file_list.append(path)
        self.transform=transform
        self.length=len(self.file_list)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        file_path=self.file_list[index]
        image=Image.open(file_path)
        image=self.transform(image)
        return image

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
    # print(labels.shape)
