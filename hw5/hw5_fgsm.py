import numpy as np
import pandas as pd
import os
import sys
from skimage.io import imread, imsave
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F

class data(Dataset):
    def __init__(self, image_dir, labels, transform = None):
        self.image_dir = image_dir
        # self.images = os.listdir(image_dir)
        self.images = [file for file in os.listdir(image_dir) if file.endswith('.png')]
        self.labels = labels
        self.transform = transform
        
        self.len = len(self.images)
          
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        img_fn = self.images[index]
        image = imread(os.path.join(self.image_dir, img_fn))
        
        if self.transform is not None:
            image = self.transform(image)
        
        idx = int(img_fn[0:3])
            
        return image, label[idx], img_fn

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
    
def load_data(label_path):
    data = pd.read_csv(label_path, header = 0)
    data = np.array(data.values)

    label = data[:, 3]
    # label.shape = (200,)
    np.save('label.npy', label)

    return label

def deprocess(adv):
    adv = adv.cpu().data.numpy()
    adv = adv.transpose(1, 2, 0)
    
    img = np.zeros(adv.shape)  # (224, 224, 3)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        img[:, :, i] = (adv[:, :, i] * std[i]) + mean[i]
    
    img = np.clip(img, 0, 1)
      
    return img

def train(train_loader, model, output_dir):
    model = model.to(device)
    model.eval()
    
    criterion = F.nll_loss
    
    epsilon = 0.3
    
    for data, target, img_fn in train_loader:
        data, target = data.to(device), target.to(device)
        
        data.requires_grad = True
        output = model(data)
        
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        
        grad = data.grad.data
        sign = grad.sign()
        adv = data + epsilon * sign
        
        for i in range(len(img_fn)):
            fn = os.path.join(output_dir, img_fn[i])
            img = deprocess(adv[i])
            imsave(fn, img)
    
if __name__ == '__main__':
    # label = load_data('labels.csv')
    label = np.load('label.npy')
    
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123564)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used: %s' % device)
        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    trainset = data(image_dir = sys.argv[1], labels = label, 
                    transform = transforms.Compose([transforms.ToTensor(), normalize]))
    
    train_loader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=1)  
    
    model = models.resnet50(pretrained=True)
    train(train_loader, model, output_dir = sys.argv[2])
