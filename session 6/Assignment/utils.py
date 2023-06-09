import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets

import matplotlib.pyplot as plt
import numpy as np


def has_cuda()->bool:
    if torch.cuda.is_available():
        return True
    else:
        return False

def which_device()->str:
    device:str = torch.device("cuda" if has_cuda() else "cpu")
    return device



def calculate_mean_std_mnist(datasets)->tuple:
    mean = 0.13065974414348602
    std = 0.3015038073062897

    # UNCOMMENT If you've decent computation

    # data_loader = DataLoader(datasets,batch_size=1,shuffle=False)
    # mean = torch.zeros(1);
    # std = torch.zeros(1)
    # num_samples = 0
    # transform = transforms.ToTensor()
    # for img in data_loader:
    #     image = img[0]
    #     image = image.squeeze()
    #     mean += image.mean()             # mean across channel sum for all pics
    #     std  += image.std()
    #     num_samples += 1
        
    # mean /= num_samples
    # std /= num_samples
    # return (mean.item(),std.item())

    return (mean,std)


# mean,std = calculate_mean_std_mnist(mnist_data)




def plot_single_mnist_img(datasets)->None:
  data_loader = DataLoader(datasets,batch_size=32,shuffle=True)
  imgs,labels   = next(iter(data_loader))
  batch_size    = imgs.size(0)
  random_number = np.random.choice( int(batch_size) , )

  print(f"shape of batch ={imgs.shape}")
  print(f"number of imgs in each batch= {len(imgs)}")
  print(f"shape of img = {imgs[random_number].shape}")

  plt.imshow(imgs[random_number].squeeze(0),  cmap='gray');
  plt.title(label=labels[random_number])
  plt.show();
