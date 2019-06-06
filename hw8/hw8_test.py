import numpy as np
import pandas as pd
import sys
import csv
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)


class data(Dataset):
    def __init__(self, x, transform = None):
        self.x = x
        self.transform = transform
        
        self.len = x.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.x[index].reshape(48, 48))
        if self.transform is not None:
            image = self.transform(image)

        return image, index

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



def load_data(test_path):
    test = pd.read_csv(test_path, header = 0)
    test = np.array(test.values)

    feature = test[:, 1]
    n = len(feature)
    x_test = np.zeros((n, 48*48))
    for i in range(n):
        x = [int(f) for f in feature[i].split()]
        x = np.array(x)
        x_test[i] = x
    x_test = x_test.reshape(n, 1, 48, 48)

    return x_test


test_path = sys.argv[1]
x_test = load_data(test_path)

test_set = data(x_test, transform = transforms.ToTensor())
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=1)


model = Net().to(device)
model.load_state_dict(torch.load('model_7.pth'))
model.eval()

result = {}
with torch.no_grad():
    for data, index in test_loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(1, keepdim=True)[1]
        pred = pred.view(pred.size(0)).data

        for i in range(len(index)):
            result[index[i].item()] = pred[i].item()


file = open(sys.argv[2], 'w+')
out_file = csv.writer(file, delimiter = ',', lineterminator = '\n')
out_file.writerow(['id', 'label'])
for i in range(len(result)):
    out_file.writerow([i, result[i]])
file.close()

