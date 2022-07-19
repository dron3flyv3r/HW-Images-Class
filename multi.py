print("importing...")
import albumentations as alb
import cv2
import os
from tqdm import tqdm
print("importing done")

classes = ["pizza", "not_pizza"]

tranform = alb.Compose([
    alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    alb.RandomCrop(300,300, p=0.5),
    alb.HorizontalFlip(p=0.5),
    alb.VerticalFlip(p=0.5),
])

for folder in classes:
    folderPath = os.path.join("data/raw", folder)
    pBar = tqdm(desc=f"Processing {folder}")
    for file in os.listdir(folderPath):
        if file.startswith("._"):
            continue
        try:
            img = cv2.imread(os.path.join(folderPath, file), cv2.COLOR_BGR2RGB)
            for i in range(15):
                tranImg = tranform(image=img)["image"]
                name = file.replace(".jpg", "").replace(".png", "")
                cv2.imwrite(os.path.join("data/multi", folder, f"{name}[{i}].jpg"), tranImg)
        except Exception as e:
            print(f"{file} failed")
            continue
        pBar.update(1)

