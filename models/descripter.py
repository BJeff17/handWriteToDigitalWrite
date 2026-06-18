import torch
from torch import nn

class Descripter(nn.Module):
    def __init__(self, n_label):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 8)
        self.pool = nn.MaxPool2d(8, 2)
        self.conv2 = nn.Conv2d(16, 8, 2)

        self.fc = nn.Sequential(
            nn.Linear(8, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_label)
        )
    
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = torch.flatten(x,1)
        return self.fc(x)
    
