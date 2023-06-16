import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,OneCycleLR
from torch.utils.data import Dataset, DataLoader


from torchvision import transforms,datasets
from torchvision.transforms import RandomPerspective,RandomRotation,RandomCrop



from torchsummary import summary
from tqdm import tqdm

SEED = 1
# For reproducibility
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)




def calculate_mnist_mean_std(data_sets):
    '''data_sets: Expected to recieve mnist dataset'''
    mean = torch.zeros(1); 
    std=torch.zeros(1);
    total_images  = torch.zeros(1)
    data_loader = DataLoader(data_sets,batch_size=128,shuffle=False)
    
    for imgs,labels in data_loader:
        batch_size = imgs.shape[0]
        total_images += batch_size
        
        imgs = imgs.view( batch_size, imgs.size(1), -1 )
        mean += imgs.mean(2).sum()
        std  +=  imgs.std(2).sum()
        
    
    mean /= total_images
    std  /= total_images
    print(f'mean of the dataset:- {mean}')
    print(f'std  of the dataset:- {std}')
    return(mean,std)

# mean, std = calculate_mnist_mean_std(mnist_data)



def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


device = 'cuda'if torch.cuda.is_available() else "cpu"



def plot_loss_accuracy(train_losses, test_losses, train_acc, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")