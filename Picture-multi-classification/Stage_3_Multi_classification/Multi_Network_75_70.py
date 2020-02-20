# import packages
import torch.nn as nn
import torch.nn.functional as F


# define class Net
class Net(nn.Module):
    # # define __init__ function
    def __init__(self):
        # ## call __init__ function from super class
        super(Net, self).__init__()
        # ## define net structure
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.bn1 = nn.BatchNorm2d(3)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(3, 6, 3)
        self.bn2 = nn.BatchNorm2d(6)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(6, 12, 3)
        self.bn3 = nn.BatchNorm2d(12)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(6 * 123 * 123, 150)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc3_1 = nn.Linear(150, 2)
        self.softmax_1 = nn.Softmax(dim=1)

        self.fc2 = nn.Linear(12 * 60 * 60, 150)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc3_2 = nn.Linear(150, 3)
        self.softmax_2 = nn.Softmax(dim=1)

    # # define forward function
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x_classes = x.view(-1, 6 * 123 * 123)
        x_classes = self.fc1(x_classes)
        x_classes = self.relu4(x_classes)
        x_classes = F.dropout(x_classes, training=self.training)
        x_classes = self.fc3_1(x_classes)
        x_classes = self.softmax_1(x_classes)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x_species = x.view(-1, 12 * 60 * 60)
        x_species = self.fc2(x_species)
        x_species = self.relu5(x_species)
        x_species = F.dropout(x_species, training=self.training)
        x_species = self.fc3_2(x_species)
        x_species = self.softmax_2(x_species)

        return x_classes, x_species
