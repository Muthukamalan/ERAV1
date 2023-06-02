import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms


def has_cuda()->bool:
    is_cuda = torch.cuda.is_available()
    print(f'CUDA Available? {is_cuda}')
    return is_cuda


def which_device()->str:
    device:str = "cpu"
    if has_cuda():
        device="cuda"
    return device


def random_image_in_batch(data_loader):
    imgs,labels = next(iter(data_loader))
    random_number:int = np.random.choice( data_loader.batch_size )   # choice random number under batch_size
    
    # Meta Imfo
    print(f"shape of each batch: {imgs.shape}")
    print(f"each batch contain {len(imgs)} images")
    print(f"shape of img = {imgs[random_number].shape}")
    
    plt.imshow(imgs[random_number].squeeze(0),  cmap='gray');
    plt.title(label=labels[random_number].item())
    return plt