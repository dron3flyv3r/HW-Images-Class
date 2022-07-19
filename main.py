from csv import writer
from genericpath import isfile
import os
from tqdm import tqdm
from Data import PizzaDataset
from model import Net
import torch
import torchvision
import sys
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter



if not os.path.exists('runs/'):
    os.makedirs('runs/')
epNum = len([entry for entry in os.listdir("runs/") if os.path.isfile(os.path.join("runs/", entry))])
writer = SummaryWriter(f"runs/ep{epNum}")

# Hyperparameters
batchSize = 32
epochs = 10
classes = ["pizza", "not_pizza"]
dataPath = r"./data/multi"
imgSize = (256, 144)

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Loading the data from the data folder and splitting it into train and test sets.
dataLoader = PizzaDataset(tranform=torchvision.transforms.Resize(imgSize), classes=classes, dataPath=dataPath)

dataLen = int(len(dataLoader))
trainLen = int(int(len(dataLoader)) * 0.8)
testLen = dataLen - trainLen
trainSet, testSet = torch.utils.data.random_split(dataLoader, [trainLen, testLen])

trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=batchSize, shuffle=False)

# Show images
exsampels = iter(trainloader).next()
samples, label = exsampels

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
#print(' '.join(f'{classes[label[j]]:5s}' for j in range(6)))
#plt.show()

writer.add_image('Sample of images', torchvision.utils.make_grid(samples))

# Architecture
def acc():
    """
    It takes the test data, feeds it to the network, and then calculates the accuracy
    :return: The accuracy of the model
    """
    correct = 0
    total = 0
    try:
        with torch.no_grad():
            for data in iter(testLoader).next():
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return (correct / total) * 100.0
    except StopIteration:
        return "No data"

# Model
net = Net().to(device)
#net = torchvision.models.googlenet(pretrained=True).to(device)
writer.add_graph(net, torch.rand(1, 3, 144, 144))

# Loss and optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.1)

# Train the model
pBar = tqdm(range(epochs), unit=" epoch")
trainBatchLen = len(trainloader)
oldAcc = 0
bestAcc = 0

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
        tBar.set_description(f"Loss: {loss.item():.4f}")
    scheduler.step()
    tmpAcc = acc()
    writer.add_scalar("Loss", running_loss/trainBatchLen, epoch + 1)
    writer.add_scalar("Lerning rate", scheduler.get_last_lr()[0], epoch + 1)
    writer.add_scalar("Accuracy", tmpAcc, epoch + 1)
    writer.add_scalar("Accuracy change", tmpAcc - oldAcc, epoch + 1)
    writer.close()
    oldAcc = tmpAcc
    if tmpAcc > bestAcc:
        bestAcc = tmpAcc
        torch.save(net.state_dict(), f"runs/ep{epNum}/best.pt")
    pBar.set_description(f'loss: {running_loss:.4f} | lerning rate: {scheduler.get_last_lr()[0]:.4f} | acc: {tmpAcc:.2f if epoch%10==0 else oldAcc}%')

torch.save(net.state_dict(), f"runs/ep{epNum}/last.pt")
print('Finished Training')