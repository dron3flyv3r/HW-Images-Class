import os
from tqdm import tqdm
from Data import PizzaDataset
from model import Net
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 16
epchs = 10

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataloader
dataloader = PizzaDataset(tranform=transforms.Resize((96, 96)))
dataLen = int(len(dataloader))
trainLen = int(int(len(dataloader)) * 0.8)
testLen = dataLen - trainLen
trainSet, testSet = random_split(dataloader, [trainLen, testLen])

trainloader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)
testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=False)

exsampels = iter(trainloader).next()
samples, label = exsampels

# Architecture
def acc():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100.0

# See a preview of the data using plt
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap="Accent")
#plt.show()

# Model
net = Net().to(device)
#net = torchvision.models.googlenet(pretrained=True).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# Train the model
pBar = tqdm(range(epchs))

for epoch in pBar:
    running_loss = 0.0
    tBar = tqdm(trainloader)
    for i, data in enumerate(tBar, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    pBar.set_description(f"loss: {running_loss}, acc: {acc()}")

print('Finished Training')

