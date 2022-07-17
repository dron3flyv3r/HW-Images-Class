from dis import dis
import os
from tqdm import tqdm
from Data import PizzaDataset
from model import Net
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyperparameters
batchSize = 16
epochs = 10

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dataloader and data split
dataLoader = PizzaDataset(tranform=transforms.Resize((96, 96)))

dataLen = int(len(dataLoader))
trainLen = int(int(len(dataLoader)) * 0.8)
testLen = dataLen - trainLen
trainSet, testSet = torch.utils.data.random_split(dataLoader, [trainLen, testLen])

trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=batchSize, shuffle=False)

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

# Model
net = Net().to(device)
#net = torchvision.models.googlenet(pretrained=True).to(device)

# Loss and optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1e-3 * (1 - x / epochs))

# Train the model
pBar = tqdm(range(epochs), unit=" epoch")

for epoch in pBar:
    running_loss = 0.0
    tBar = tqdm(trainloader, unit=" batch", leave=False)
    dataLen
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
        tBar.set_description(f"Loss: {running_loss / (i + 1):.3f}")
    tBar.reset()
    scheduler.step()
    pBar.set_description(f'loss: {running_loss}, acc: {acc()}')

print('Finished Training')