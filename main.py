import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets
from torchvision import transforms as T
from auto_encoder import AutoEncoder


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--in-dir', type=str, default='./data')
parser.add_argument('-o', '--out-dir', type=str, default='./output')
parser.add_argument('-b', '--batch-size', type=int, default=128)
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-lr', type=float, default=1e-4)
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
dcount = max(1, torch.cuda.device_count())

args.batch_size *= dcount
kwargs = {
    'num_workers': os.cpu_count(),
    'pin_memory': True
    } if use_cuda else {}

dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(args.in_dir, train=True, download=True, transform=T.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

model = AutoEncoder().to(device)
if dcount > 1:
    model = nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
model.train()

optimizer = optim.Adam(model.parameters(), lr=args.lr*dcount)
criterion = nn.BCELoss(reduction='sum')

for epoch in range(args.epochs):
    total_loss = 0
    for img, _ in tqdm(dataloader):
        img = img.to(device)

        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    print(f'Epoch: {epoch} Loss: {total_loss/len(dataloader.dataset):.4f}')

os.makedirs(args.out_dir, exist_ok=True)
torch.save(model.state_dict(), f'{args.out_dir}/latest.pth')
print('Save model weights')

