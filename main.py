import os
from tqdm import tqdm
from Data import PizzaDataset
from model import Net
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys

if not os.path.exists('runs/'):
    os.makedirs('runs/')
epNum = len(next(os.walk('runs/'))[1])
while os.path.exists(f"runs/ep{epNum}"):
    epNum += 1
    if not os.path.exists(f"runs/ep{epNum}"):
        break
writer = SummaryWriter(f"runs/ep{epNum}")

# Hyperparameters
batchSize = 32
epochs = 25
nClasses = 2
classes = ["pizza", "not_pizza"]
dataPath = r"./data/multi"
imgSize = (224, 224)
std = [0.2059, 0.2202, 0.2091]
mean = [0.5657, 0.4289, 0.3180]

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Loading the data from the data folder and splitting it into train and test sets.
trainTransform = transforms.Compose([
                                transforms.Resize(imgSize),
                                transforms.ToTensor(),
                                transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
                                ])
dataLoader = PizzaDataset(tranform=trainTransform, classes=classes, dataPath=dataPath)

dataLen = int(len(dataLoader))
trainLen = int(int(len(dataLoader)) * 0.8)
testLen = dataLen - trainLen
trainSet, testSet = torch.utils.data.random_split(dataLoader, [trainLen, testLen])

trainloader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=batchSize, shuffle=False)

def meen(loader):
    mean = 0.
    std = 0.
    for images, _ in tqdm(loader):
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    print(f"std: {std} | mean {mean}")


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
            for data in testLoader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                break
        return (correct / total) * 100.0
    except StopIteration:
        return "No data"

def saveCheckpoint(model, optimizer, acc, classes, trans=None, filename='checkpoint'):
    state = {
        'epoch': epoch + 1,
        'modelState': model.state_dict(),
        'model': model,
        'macModel': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'accuracy': acc,
        'classes': classes,
        'transform': trans
    }
    torch.save(state, f"runs/ep{epNum}/{filename}.pth.tar")

# Model
#net = Net().to(device)
#net = torchvision.models.googlenet(pretrained=True).to(device)
net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
nFeet = net.fc.in_features
net.fc = nn.Linear(nFeet, nClasses)
net = net.to(device)

# Loss and optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25)

# Train the model
pBar = tqdm(range(epochs), unit=" epoch")
trainBatchLen = len(trainloader)
oldAcc = 0
bestAcc = 0

for epoch in pBar:
    running_loss = 0.0
    tBar = tqdm(trainloader, unit=" batch", leave=False)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # zero the parameter gradients
        optimizer.zero_grad()

        running_loss += loss.item()
        tBar.set_description(f"Loss: {loss.item():.4f}")
        tBar.update(1)
    scheduler.step()
    tmpAcc = acc()
    writer.add_scalar("Loss", running_loss/trainBatchLen, epoch + 1)
    writer.add_scalar("Lerning rate", scheduler.get_last_lr()[0], epoch + 1)
    writer.add_scalar("Accuracy", tmpAcc, epoch + 1)
    writer.add_scalar("Accuracy change", tmpAcc - oldAcc, epoch + 1)

    oldAcc = tmpAcc
    if tmpAcc > bestAcc:
        bestAcc = tmpAcc
        torch.save(net, f"runs/ep{epNum}/best.pt")
        torch.save(net.state_dict(), f"runs/ep{epNum}/mac.pt")
        saveCheckpoint(net, optimizer, bestAcc, classes, trainTransform, filename="best")

    pBar.set_description(f'loss: {running_loss:.4f} | acc: {tmpAcc:.2f}% | best acc: {bestAcc:.2f}% | lerning rate: {scheduler.get_last_lr()[0]:.4f}')
writer.close()
torch.save(net, f"runs/ep{epNum}/last.pt")
saveCheckpoint(net, optimizer, oldAcc, classes, trainTransform, filename="last")
print('Finished Training')
print(f"Best accuracy: {bestAcc}%")
print(f"Last accuracy: {tmpAcc}%")