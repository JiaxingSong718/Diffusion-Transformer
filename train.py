from config import *
from torch.utils.data import DataLoader
from dataset import MNIST
from diffusion import forward_add_noise
import torch 
from torch import nn 
import os
from tqdm import tqdm
from Diffusion_Transformer import DiffusionTransformer

DEVICE='cuda' if torch.cuda.is_available() else 'cpu' 

dataset=MNIST()

model=DiffusionTransformer(img_size=28,patch_size=4,channel=1,emb_size=64,label_num=10,dit_num=3,head=4).to(DEVICE)

try: 
    model.load_state_dict(torch.load('./checkpoints/DiT_model.pth'))
except:
    pass 

optimzer=torch.optim.Adam(model.parameters(),lr=1e-3) 
loss_fn=nn.L1Loss()

EPOCH=500
BATCH_SIZE=300
best_loss = float('inf')

dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)

model.train()
iter_count=0
for epoch in range(EPOCH):
    last_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCH}", unit="batch",ncols=200) 
    for imgs,labels in progress_bar:
        x=imgs*2-1 # 图像的像素范围从[0,1]转换到[-1,1],和噪音高斯分布范围对应
        t=torch.randint(0,T,(imgs.size(0),))  # 为每张图片生成随机t时刻
        y=labels
        
        x,noise=forward_add_noise(x,t) # x:加噪图 noise:噪音
        pred_noise=model(x.to(DEVICE),t.to(DEVICE),y.to(DEVICE))

        loss=loss_fn(pred_noise,noise.to(DEVICE))
        
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        last_loss = loss.item()
        # if iter_count%1000==0:
        #     print('epoch:{} iter:{},loss:{}'.format(epoch,iter_count,loss))
        iter_count+=1
        progress_bar.set_postfix(loss=last_loss)

    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(),'./checkpoints/DiT_model.pth')
        os.replace('./checkpoints/DiT_model.pth','./checkpoints/DiT_model.pth')
        print('Model saved with loss:', best_loss)