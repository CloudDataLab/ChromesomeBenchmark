import pretrainedmodels
import torch
import torch.nn as nn

__all__ = ['CNN', 'DeepCNN', 'Vanilla', 'ChromeNet2', 'CirNet', 'MixNet']


class CNN(nn.Module):
    def __init__(self, n_classes=24):
        super(CNN, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout2d(0.25),

            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            nn.Flatten(),
            nn.Linear(30976, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes))

    def forward(self, inputs):
        x = self.model(inputs)
        return x


class DeepCNN(nn.Module):
    def __init__(self, n_classes=24):
        super(DeepCNN, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25),

            nn.Conv2d(16, 8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25))
        self.fc = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(72, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes))

    def forward(self, inputs):
        x = self.model(inputs)
        x = self.fc(x)
        return x


class ChromeNet2(nn.Module):
    def __init__(self, n_classes=24):
        super(ChromeNet2, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.Dropout2d(0.3),

            nn.Flatten(),
            nn.Linear(15488, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes))

    def forward(self, inputs):
        x = self.model(inputs)
        return x


class Vanilla(nn.Module):
    def __init__(self, n_classes=24):
        super(Vanilla, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, ceil_mode=True),
            nn.Dropout2d(0.5),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, ceil_mode=True),

            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.Dropout2d(0.5))
        self.fc = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(61952, 120),
            nn.ReLU(),
            nn.Linear(120, n_classes))

    def forward(self, inputs):
        x = self.model(inputs)
        x = self.fc(x)
        return x


class MixNet(nn.Module):
    def __init__(self, n_classes=24):
        super(MixNet, self).__init__()
        self.model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        num_fc_ftr = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fc_ftr, n_classes)

    def forward(self, inputs):
        x = self.model(inputs)
        return x


class CirNet(nn.Module):
    def __init__(self, n_classes=24):
        super(CirNet, self).__init__()
        self.model = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=n_classes, pretrained=None)

    def forward(self, inputs):
        x = self.model(inputs)
        return x
