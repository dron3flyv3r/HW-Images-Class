import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = None
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x.float())), 2)
        x = F.max_pool2d(F.relu(self.conv2(x.float())), 2)

        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 512)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x