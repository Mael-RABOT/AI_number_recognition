import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import gradio
import model.NeuralNetwork as model
from time import sleep

from torchvision import transforms
from tqdm import tqdm
from PIL import Image

class app:
    def __init__(self, epoch=10, batch_size=64, learning_rate=0.01):
        self.model = model.NeuralNetwork()
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_set = None
        self.test_set = None
        self.train_loader = None

    def add_dataset(self, dataset_path):
        self.train_set = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True, transform=transforms.ToTensor())
        self.test_set = torchvision.datasets.MNIST(root=dataset_path, train=False, download=True, transform=transforms.ToTensor())
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        assert len(self.train_loader) == 938, "Your training dataset of 60,000 images isn't split with a 64 batch_size"
        return self.train_loader

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model has been saved")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print("Model has been loaded")

    def predict_image(self, input):
        return self.model.forward(input)

    def launch_app(self, gradio_server=False, load_dataset=False, load_model=True, show_log=False, share=False):
        if load_dataset:
            self.add_dataset("./data/dataset")
        if not load_model:
            self.add_dataset("./data/dataset")
            self.model.train(self.epoch, self.learning_rate, self.train_loader, show_loss=show_log)
            self.model.test(self.test_set)
        else:
            self.load_model("model/model_save")
        self.save_model("model/model_save")

        IMG_SIZE = 28 if torch.cuda.is_available() else 28
        COMPOSED_TRANSFORMERS = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ])

        def reformat_image(image):
            image = Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE))
            image = COMPOSED_TRANSFORMERS(image).unsqueeze(0)
            return image.to('cpu', torch.float)

        def execute_predict(Image):
            tensor = reformat_image(Image)
            return (self.model.forward(tensor).argmax(dim=1).item())

        if gradio_server:
            SketchPad_Size = 28 * 10
            SketchPad = gradio.Sketchpad().style(height=SketchPad_Size, width=SketchPad_Size)
            demo = gradio.Interface(
                fn=execute_predict,
                inputs=SketchPad,
                outputs="label",
                title="AI for number recognition",
                description="Try writing a number in the sketchpad\nUse all the space to increase AI accuracy"
            )
            demo.launch(share=share)

App = app(epoch=10)
App.launch_app(load_model=True, show_log=False, gradio_server=True, share=True)
