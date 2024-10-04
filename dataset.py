import torch
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor,Compose
from torchvision import transforms
import torchvision

# 手写数字
class MNIST(Dataset):
    def __init__(self,is_train=True):
        super().__init__()
        self.ds=torchvision.datasets.MNIST('./mnist/',train=is_train,download=True)
        self.img_convert=Compose([
            PILToTensor(),
        ])
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self,index):
        img,label=self.ds[index]
        return self.img_convert(img)/255.0,label

#Tensor转PIL图像
tensor_to_pil = transforms.Compose([
    transforms.Lambda(lambda t: t*255),  #像素值还原
    transforms.Lambda(lambda t: t.type(torch.uint8)),  #像素值取整
    transforms.ToPILImage() #Tensor图像转为PIL, (C,H,W) -> (H,W,C)

])