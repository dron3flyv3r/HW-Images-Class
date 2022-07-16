import torch
import os
import torchvision.io as io

class PizzaDataset(torch.utils.data.Dataset):
    """Some Information about PizzaDataset"""
    def __init__(self, tranform=None):
        super(PizzaDataset, self).__init__()
        dataPath = ["data/pizza", "data/not_pizza"]
        label = 0
        self.data = []
        self.tranform = tranform
        for path in dataPath:
            for file in os.listdir(path):
                if file.startswith("._") and file.endswith(".jpg") or file.endswith(".png"):
                    continue
                tmpData = [os.path.join(path, file), label]
                self.data.append(tmpData)
            label += 1

    def __getitem__(self, index):
        img = io.read_image(self.data[index][0], io.ImageReadMode.GRAY).type(torch.float32)
        if self.tranform:
            img = self.tranform(img)
        label = torch.tensor(int(self.data[index][1]))
        return (img, label)

    def __len__(self):
        return len(self.data)