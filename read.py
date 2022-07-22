import torch
import torchvision.transforms as transforms
from PIL import Image

classes = ["pizza", "not_pizza"]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
imgSize = (224, 224)
std = [0.2059, 0.2202, 0.2091]
mean = [0.5657, 0.4289, 0.3180]

model = torch.load(r"runs\ep6\best.pt")
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize(imgSize),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)
)])

def classify(model, imageTransform, imgPath, classes):
    model = model.eval()
    img = Image.open(imgPath)
    img = imageTransform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    output = model(img)
    _, pred = torch.max(output, 1)
    print(f"Prediction: {classes[pred.item()]}")

classify(model, transform, r"test\bella.jpg", classes)