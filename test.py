import torch
from PIL import Image

checkpoint = torch.load(r"runs\ep6\best.pth.tar")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def classify(model, imgPath, trans=None, classes=[]):
    model = model.eval()
    img = Image.open(imgPath)
    img = trans(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    output = model(img)
    _, pred = torch.max(output, 1)
    print(f"Prediction: {classes[pred.item()]}")

imgPath = r"test\pizza.jpg"
model = checkpoint['model']
classes = checkpoint['classifier']
tran = checkpoint['transform']
classify(model, imgPath, tran, classes)

