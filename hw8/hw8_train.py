import numpy as np
import pandas as pd
import sys
import csv
import time
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)


class data(Dataset):
    def __init__(self, x, y, transform = None):
        self.x = x
        self.y = y
        self.transform = transform
        
        self.len = x.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.x[index].reshape(48, 48))
        if self.transform is not None:
            image = self.transform(image)

        return image, self.y[index]

    def __len__(self):
        return self.len


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(1, 32, 1),
            conv_bn(32, 32, 1),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            conv_dw(32, 64, 1),
            conv_dw(64, 64, 1),
            conv_dw(64, 80, 2),
            conv_dw(80, 80, 1),
            conv_dw(80, 80, 2),
            conv_dw(80, 80, 1),
            nn.AvgPool2d(6),
            nn.Dropout(0.5)
        )

        self.fc = nn.Sequential(
            nn.Linear(80, 7),
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 80)
        x = self.fc(x)
        return x


def load_data(train_path):
    train = pd.read_csv(train_path, header = 0)
    train = np.array(train.values)

    Y_train = train[:, 0]

    feature = train[:, 1]
    n = len(feature)
    X_train = np.zeros((n, 48*48))
    for i in range(n):
        x = [int(f) for f in feature[i].split()]
        x = np.array(x)
        X_train[i] = x
    X_train = X_train.reshape(n, 1, 48, 48)

    x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 0)
    
    return x_train, y_train, x_valid, y_valid


train_path = sys.argv[1]
x_train, y_train, x_valid, y_valid = load_data(train_path)


train_set = data(x_train, y_train, 
        transform = transforms.Compose([ 
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])
    )

valid_set = data(x_valid, y_valid, 
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    )

batch_size = 128
lr = 0.0005
n_epoch = 150

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=1)


model = Net().to(device)
# print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
best_acc = 0.0

for epoch in range(n_epoch):
    epoch_start_time = time.time()

    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        output = model(data[0].to(device))
        loss = criterion(output, data[1].to(device))
        loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += loss.item()

    model.eval()
    for i, data in enumerate(valid_loader):
        pred = model(data[0].to(device))
        loss = criterion(pred, data[1].to(device))

        val_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        val_loss += loss.item()

    val_acc = val_acc / len(valid_set)
    train_acc = train_acc / len(train_set)
    val_loss = val_loss / len(valid_set)
    train_loss = train_loss / len(train_set)
    
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f'
        % (epoch + 1, n_epoch, time.time()-epoch_start_time, train_acc, train_loss, val_acc, val_loss))

    if (val_acc > best_acc):
        torch.save(model.state_dict(), 'my_model.pth')
        best_acc = val_acc
        print ('Model Saved!')

