import torch
import os
import torchvision.io as io

class PizzaDataset(torch.utils.data.Dataset):
    """Some Information about PizzaDataset"""
    def __init__(self, tranform=None, classes=[], dataPath=""):
        super(PizzaDataset, self).__init__()
        label = 0
        self.data = []
        self.tranform = tranform
        for cla in classes:
            path = os.path.join(dataPath, cla)
            for file in os.listdir(path):
                if file.startswith("._") and (not file.endswith(".jpg") or not file.endswith(".png")):
                    continue
                tmpData = [os.path.join(path, file), label]
                self.data.append(tmpData)
            label += 1

    def __getitem__(self, index):
        img = io.read_image(self.data[index][0])
        if self.tranform:
            img = self.tranform(img)
        label = torch.tensor(int(self.data[index][1]))
        return (img, label)

    def __len__(self):
        return len(self.data)