from torch import nn as nn
from torch.nn import functional as F

class Net2(nn.Module):
    def __init__(self)->None:
        super(Net2,self).__init__()

        self.conv1 = nn.Sequential(
            # Input 1 channels,  output=16 channels
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1,bias=False),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Input 16 Channels, outputs=32 channels
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1,bias=False), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.trans1 = nn.Sequential(
            # Input 32 channels output=16 channels
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,bias=False,padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Input resolution shape = (28*28*16)   output = (14*14*16)
            nn.MaxPool2d( kernel_size =2 , stride =2 , padding =0 )
        )
        
        self.conv2 =nn.Sequential(
            # input= 16 channels, output=(16 channels)
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,bias=False), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # input 16 channels and output=(32 channels)
            nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False),  
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.trans2 = nn.Sequential(
            # input 32 channels and output = 16 channels
            nn.Conv2d(in_channels=64,out_channels=16,kernel_size=1,bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(16),

            # input = (14*14*16) output=(7*7*16)
            nn.MaxPool2d( kernel_size =2 , stride =2 , padding =0 )
        )

        self.conv3 = nn.Sequential(
            # input=(7*7*16)   output=(5*5*10)
            nn.Conv2d(in_channels=16 ,out_channels=10, kernel_size=3,stride=1,padding=0,bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3)  #(1*1*10)
        )
        
        
    def forward(self,x):
        x = self.trans1( self.conv1(x) )
        x = self.trans2( self.conv2(x) )
        x = self.conv3(x)
        x = x.view(-1,10)
        return F.log_softmax(x)