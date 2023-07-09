import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net256_Conv5_Fc3_B_C7(nn.Module):
    def __init__(self):
        super().__init__()

        self.to_linear = 200 * 4 * 4
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 160, 5)
        self.conv5 = nn.Conv2d(160, 200, 5)

        self.fc1 = nn.Linear(self.to_linear, 512)
        self.fc2 = nn.Linear(512, 200)
        self.fc3 = nn.Linear(200, 7)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = x.view(-1, self.to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)


class Net256_Conv5_Fc3_B_C2(nn.Module):
    def __init__(self):
        super().__init__()

        self.to_linear = 200 * 4 * 4
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 160, 5)
        self.conv5 = nn.Conv2d(160, 200, 5)

        self.fc1 = nn.Linear(self.to_linear, 512)
        self.fc2 = nn.Linear(512, 200)
        self.fc3 = nn.Linear(200, 2)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = x.view(-1, self.to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)


class Net256_Conv5_Fc3_B_RGB_C2(nn.Module):
    def __init__(self):
        super().__init__()

        self.to_linear = 200 * 4 * 4
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 160, 5)
        self.conv5 = nn.Conv2d(160, 200, 5)

        self.fc1 = nn.Linear(self.to_linear, 512)
        self.fc2 = nn.Linear(512, 200)
        self.fc3 = nn.Linear(200, 2)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = x.flatten(start_dim=1, end_dim=- 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)


class Net256_Conv5_Fc3_B_RGB_C7(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_linear = 200 * 4 * 4
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 160, 5)
        self.conv5 = nn.Conv2d(160, 200, 5)

        self.fc1 = nn.Linear(3200, 512)
        self.fc2 = nn.Linear(512, 200)
        self.fc3 = nn.Linear(200, 7)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        #x = x.view(-1, self.to_linear)
        x = x.flatten(start_dim=1, end_dim=- 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)
