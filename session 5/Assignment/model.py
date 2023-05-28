import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary



class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,bias=False)   
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,bias=False)   
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3,bias=False)
        self.fc1 = nn.Linear(4096, 50,bias=False)
        self.fc2 = nn.Linear(50, 10,bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)                 # In: 28*28*1 | Out: 26*26*32 | RF: 1>3
        x = F.relu(F.max_pool2d(self.conv2(x), 2))   # In: 26*26*32 > 24*24*64 > 12*12*64  | RF: 3>5>6
        x = F.relu(self.conv3(x), 2)                 # In: 12*12*64 | Out: 10*10*128 | RF: 6>10
        x = F.relu(F.max_pool2d(self.conv4(x), 2))   # In: 10*10*128 > 8*8*256 >  4*4*256 | RF: 10>14>16
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

class Net2(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,bias=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,bias=True)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3,bias=True)
        self.fc1 = nn.Linear(4096, 50,bias=True)
        self.fc2 = nn.Linear(50, 10,bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) 
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    


    
def model_summary(model,input_size):
    description = summary(model, input_size)
    return description