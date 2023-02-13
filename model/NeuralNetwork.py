import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import gradio

from torchvision import transforms
from tqdm import tqdm
from PIL import Image

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=5)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2,
                                           stride=2)
        self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=5)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2,
                                           stride=2)
        self.linear1 = torch.nn.Linear(16, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        x = x.reshape(-1, 4 * 4)

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

    def train(self, EPOCH, learning_rate, train_loader, show_loss=False):
        loss_fonct = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.01)

        for epoch in tqdm(range(EPOCH), ascii=' ='):
            for image, label in tqdm(train_loader, ascii=' ='):
                pred = self.forward(image)
                loss = loss_fonct(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if show_loss:
                print(round(loss.item(), 2))

    def test(self, test_set):
        total, correct = 0, 0
        for image, label in tqdm(test_set):
            output = self.forward(image.reshape(1, 1, 28, 28))
            if (output.argmax(dim=1).item() == label):
                correct += 1
            total += 1

        accuracy = correct / total * 100
        print(f'Your accuracy is {accuracy}% !')
        assert accuracy > 80, "Your accuracy is not good enough, keep trying to build"