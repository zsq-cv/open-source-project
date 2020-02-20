import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.bn1 = nn.BatchNorm2d(3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, 6, 3)
        self.bn2 = nn.BatchNorm2d(6)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(6, 12, 3)
        self.bn3 = nn.BatchNorm2d(12)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(12, 24, 3)
        self.bn4 = nn.BatchNorm2d(24)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(24 * 29 * 29, 150)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(150, 3)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.maxpool3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.maxpool4(x)
        x = self.relu4(x)

        x = x.view(-1, 24 * 29 * 29)
        x = self.fc1(x)
        x = self.relu6(x)

        x = F.dropout(x, training=self.training)

        x_species = self.fc2(x)
        x_species = self.softmax1(x_species)

        return x_species
