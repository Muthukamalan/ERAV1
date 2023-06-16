

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




# FULLY CONNECTED LAYER NETWORK
class Net1(nn.Module):
    def __init__(self)->None:
        super(Net1,self).__init__()
        self.fc1 = nn.Linear(in_features=28*28,out_features=128)
        self.fc2 = nn.Linear(in_features=128,out_features=64)
        self.fc3 = nn.Linear(in_features=64,  out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=10)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = x.view( x.size(0) , -1 )
        x = self.relu( self.fc1(x) )
        x = self.relu( self.fc2(x) )
        x = self.relu( self.fc3(x) )
        x = self.fc4(x)
        return F.log_softmax(x)


# CONV + FC
class Net2(nn.Module):
    def __init__(self)->None:
        super(Net2,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,bias=False,stride=1,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,bias=False,stride=1,kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,bias=False,stride=1,kernel_size=3)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(kernel_size=2,stride=2)
        self.avg   = nn.AvgPool2d(kernel_size=3)
        self.fc1   = nn.Linear(in_features=128,out_features=64)
        self.fc2   = nn.Linear(in_features=64,out_features=32)
        self.fc3   = nn.Linear(in_features=32,out_features=10)

    def forward(self,x):
        x = self.relu( self.conv1(x)  )  # 28*28*1   > 26*26*32         #k=(3*3*32)
        x = self.pool(x)                 # 26*26*32  > 13*13*32
        x = self.relu( self.conv2(x) )   # 13*13*32  > 11*11*64        #k=(3*3*32 * 64)
        x = self.pool(x)                 # 11*11*64  > 5*5*54
        x = self.relu( self.conv3(x) )   # 5*5*64    > 3*3*128      #k=(3*3*64 *128)
        x = self.avg( x )
        x = x.view( x.size(0) , -1 )
        x = self.fc3( self.relu( self.fc2( self.relu( self.fc1(x) ) ) ) )
        return F.log_softmax(x)


# FULLY CONV GAP - 1*1 Kernel Prediction Layer
class Net3(nn.Module):
    def __init__(self)->None:
        super(Net3,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,bias=False,stride=1,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,bias=False,stride=1,kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,bias=False,stride=1,kernel_size=3)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(kernel_size=2,stride=2)
        self.avg   = nn.AvgPool2d(kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=10,bias=False,kernel_size=1)

    def forward(self,x):
        x = self.relu( self.conv1(x)  )  # 28*28*1   > 26*26*32         #k=(3*3*32)
        x = self.pool(x)                 # 26*26*32  > 13*13*32
        x = self.relu( self.conv2(x) )   # 13*13*32  > 11*11*64        #k=(3*3*32 * 64)
        x = self.pool(x)                 # 11*11*64  > 5*5*54
        x = self.relu( self.conv3(x) )   # 5*5*64    > 3*3*128      #k=(3*3*64 *128)
        x = self.avg( x )                # 3*3*128   > 1*1*128
        x = self.conv4(x)
        x = x.view(x.size(0),-1)
        return F.log_softmax(x)



# FULLY CONV - GAP - Prediction Layer
class Net4(nn.Module):
    def __init__(self)->None:
        super(Net4,self).__init__()
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(kernel_size=2,stride=2)
        self.avg   = nn.AdaptiveAvgPool2d(1)


        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,bias=False,stride=1,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,bias=False,stride=1,kernel_size=3,padding=1)
        self.tran1 = nn.Conv2d(in_channels=32,out_channels=16,bias=False,stride=1,kernel_size=1,padding=1)

        self.conv3 = nn.Conv2d(in_channels=16,out_channels=32,bias=False,stride=1,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=32,out_channels=64,bias=False,stride=1,kernel_size=3,padding=1)
        self.tran2 = nn.Conv2d(in_channels=64,out_channels=16,bias=False,stride=1,kernel_size=1,padding=1)

        self.conv5 = nn.Conv2d(in_channels=16,out_channels=10,bias=False,stride=1,kernel_size=3,padding=1)


    def forward(self,x):
        x = self.pool( self.tran1( self.relu( self.conv2( self.relu( self.conv1(x) ) ) )  ) )
        x = self.pool( self.tran2( self.relu( self.conv4( self.relu( self.conv3(x) ) ) )  ) )
        x = self.conv5(x)
        x = self.avg( x )
        x = x.view(x.size(0),-1)
        return F.log_softmax(x)


# Fully COnv - 1*1 - GAP
class Net5(nn.Module):
    def __init__(self)->None:
        super(Net5,self).__init__()
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(kernel_size=2,stride=2)
        self.avg   = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,bias=False,stride=1,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=64,bias=False,stride=1,kernel_size=3,padding=1)
        self.tran1 = nn.Conv2d(in_channels=64,out_channels=32,bias=False,stride=1,kernel_size=1,padding=1)

        self.conv3 = nn.Conv2d(in_channels=32,out_channels=128,bias=False,stride=1,kernel_size=3,padding=1)
        self.tran2 = nn.Conv2d(in_channels=128,out_channels=64,bias=False,stride=1,kernel_size=1,padding=1)

        self.conv5 = nn.Conv2d(in_channels=64,out_channels=10,bias=False,stride=1,kernel_size=3,padding=1)


    def forward(self,x):
        x = self.pool( self.tran1( self.relu( self.conv2( self.relu( self.conv1(x) ) ) )  ) )
        x = self.pool( self.tran2(  self.relu( self.conv3(x) ) ) )
        x = self.conv5(x)
        x = self.avg( x )
        x = x.view(x.size(0),-1)
        return F.log_softmax(x)





# Strectch and Squeeze
class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        self.dropout_rate = 0.1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),# 28*28*1 >26*26*8 RF = 3
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),# 26*26*8 > 24*24*10, RF = 5
            nn.ReLU(),

        )
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),  # 24*24*10
            nn.MaxPool2d(kernel_size=(2,2)),  # 24*24*10 > 12*12*10
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False), #Op_size = 12*12*10 > 10*10*12
            nn.ReLU(),

            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), #10*10*12 > 8*8*14
            nn.ReLU(),

            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),#8*8*14 > 6*6*16
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),#6*6*16 > 4*4*16
            nn.ReLU(),
        )
        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),    # 4*4*16 > 4*4*10
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.trans1( self.conv1(x) )
        x = self.trans2( self.conv2(x) )
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)





# ADDED BATCHNORM
class Net7(nn.Module):
    def __init__(self):
        super(Net7, self).__init__()
        self.dropout_rate = 0.1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),# 28*28*1 >26*26*8 RF = 3
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),# 26*26*8 > 24*24*10, RF = 5
            nn.BatchNorm2d(10),
            nn.ReLU(),

        )
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),  # 24*24*10
            nn.MaxPool2d(kernel_size=(2,2)),  # 24*24*10 > 12*12*10
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False), #Op_size = 12*12*10 > 10*10*12
            nn.BatchNorm2d(12),
            nn.ReLU(),

            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), #10*10*12 > 8*8*14
            nn.BatchNorm2d(14),
            nn.ReLU(),

            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),#8*8*14 > 6*6*16
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),#6*6*16 > 4*4*16
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),    # 4*4*16 > 4*4*10
        )
        self.gap = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x = self.trans1( self.conv1(x) )
        x = self.trans2( self.conv2(x) )
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



# ADDED 0.1 DROPOUT NOT USEFUL
class Net8(nn.Module):
    def __init__(self):
        super(Net8, self).__init__()
        self.dropout_rate = 0.1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),# 28*28*1 >26*26*8 RF = 3
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),

            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),# 26*26*8 > 24*24*10, RF = 5
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),

        )
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),  # 24*24*10
            nn.MaxPool2d(kernel_size=(2,2)),  # 24*24*10 > 12*12*10
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False), #Op_size = 12*12*10 > 10*10*12
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),

            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), #10*10*12 > 8*8*14
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),

            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),#8*8*14 > 6*6*16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),#6*6*16 > 4*4*16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),
        )
        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),    # 4*4*16 > 4*4*10
        )
        self.gap = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x = self.trans1( self.conv1(x) )
        x = self.trans2( self.conv2(x) )
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


# Net_81 is same as Net_8 but less Dropouts
# USEFUL 0.01 DROPOUT
class Net8_1(nn.Module):
    def __init__(self):
        super(Net8_1, self).__init__()
        self.dropout_rate = 0.01
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),# 28*28*1 >26*26*8 RF = 3
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),

            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),# 26*26*8 > 24*24*10, RF = 5
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),
        )
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),  # 24*24*10
            nn.MaxPool2d(kernel_size=(2,2)),  # 24*24*10 > 12*12*10
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False), #Op_size = 12*12*10 > 10*10*12
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),

            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), #10*10*12 > 8*8*14
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),

            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),#8*8*14 > 6*6*16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),#6*6*16 > 4*4*16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),
        )
        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),    # 4*4*16 > 4*4*10
        )
        self.gap = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x = self.trans1( self.conv1(x) )
        x = self.trans2( self.conv2(x) )
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)




# INCREASED FILTER IN LAST LAYER
class Net9(nn.Module):
    def __init__(self):
        super(Net9, self).__init__()
        self.dropout_rate = 0.01
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),# 28*28*1 >26*26*8 RF = 3
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),
            
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),# 26*26*8 > 24*24*10, RF = 5
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),
            
        )
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),  # 24*24*10
            nn.MaxPool2d(kernel_size=(2,2)),  # 24*24*10 > 12*12*10
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False), #Op_size = 12*12*10 > 10*10*12 
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),
    
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), #10*10*12 > 8*8*14
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),
            
            nn.Conv2d(in_channels=14, out_channels=31, kernel_size=(3, 3), padding=1, bias=False),#8*8*14 > 6*6*16
            nn.BatchNorm2d(31),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),
        )
        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels=31, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),    # 4*4*16 > 4*4*10
        )   
        self.gap = nn.AdaptiveAvgPool2d(1)
        

    def forward(self, x):
        x = self.trans1( self.conv1(x) )
        x = self.trans2( self.conv2(x) )
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    




# SAME AS NET9 BUT SMALL INCREASE  in CONV1 FILTER INCREASE 10->11
# ABLE TO HIT 99.39
class Net9_3(nn.Module):
    def __init__(self):
        super(Net9_3, self).__init__()
        self.dropout_rate = 0.01
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),# 28*28*1 >26*26*8 RF = 3
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),
            
            nn.Conv2d(in_channels=8, out_channels=11, kernel_size=(3, 3), padding=0, bias=False),# 26*26*8 > 24*24*10, RF = 5
            nn.BatchNorm2d(11),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),
            
        )
        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),  # 24*24*10
            nn.MaxPool2d(kernel_size=(2,2)),  # 24*24*10 > 12*12*10
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False), #Op_size = 12*12*10 > 10*10*12 
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),
    
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), #10*10*12 > 8*8*14
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),
            
            nn.Conv2d(in_channels=14, out_channels=31, kernel_size=(3, 3), padding=1, bias=False),#8*8*14 > 6*6*16
            nn.BatchNorm2d(31),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),
            
            # nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),#6*6*16 > 4*4*16
            # nn.BatchNorm2d(10),
            # nn.ReLU(), 
            # nn.Dropout2d(self.dropout_rate),
        )
        self.trans2 = nn.Sequential(
            nn.Conv2d(in_channels=31, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),    # 4*4*16 > 4*4*10
        )   
        self.gap = nn.AdaptiveAvgPool2d(1)
        

    def forward(self, x):
        x = self.trans1( self.conv1(x) )
        x = self.trans2( self.conv2(x) )
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)




