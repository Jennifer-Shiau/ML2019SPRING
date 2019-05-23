import numpy as np
from PIL import Image
import pandas as pd
import os
import csv
import sys

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3*32*32, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 3*32*32),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 3, 32, 32)
        return x


use_cuda = torch.cuda.is_available()
torch.manual_seed(123564)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used: %s' % device)


img_dir = sys.argv[1]
transform = transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

n_imgs = 40000
X = torch.zeros(n_imgs, 3, 32, 32)

for i in range(n_imgs):
    img_fn = os.path.join(img_dir, str(i+1).zfill(6)+'.jpg')
    img = Image.open(img_fn).convert('RGB')
    img = transform(img).view(1, 3, 32, 32)
    X[i, :, :, :] = img

X = X.view(n_imgs, 3*32*32)
X = X.to(device)

model = autoencoder().to(device)
model.load_state_dict(torch.load('autoencoder.pth'))
model.eval()


code = model.encoder(X).view(n_imgs, -1)
code = code.detach().cpu().numpy()

pca = PCA(n_components=128, whiten=True, random_state=0).fit_transform(code)

kmeans = KMeans(n_clusters=2, random_state=0).fit(pca)
label = kmeans.labels_

test_file = sys.argv[2]
output_file = sys.argv[3]

data = pd.read_csv(test_file, header = 0)
data = np.array(data.values)
idx1 = data[:, 1]
idx2 = data[:, 2]

file = open(output_file, 'w+')
out_file = csv.writer(file, delimiter = ',', lineterminator = '\n')
out_file.writerow(['id', 'label'])
for i in range(data.shape[0]):
    if label[idx1[i]-1] == label[idx2[i]-1]:
        out_file.writerow([i, 1])
    else:
        out_file.writerow([i, 0])
file.close()

