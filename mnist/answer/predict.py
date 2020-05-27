from __future__ import print_function
import sys
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from main import Net, MnistDataset

def test(model, device, img):
    model.eval()
    with torch.no_grad():
        img = img.to(device)
        output = model(img.unsqueeze(0))
        predict = torch.max(output, 1)[1]

    print("This picture is {}.".format(predict.data[0]))

def main():
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    img = tfms(Image.open(sys.argv[1]))

    model = Net().to(device)
    
    model.load_state_dict(torch.load("mnist_cnn.pt"))

    test(model, device, img)

if __name__ == '__main__':
    main()