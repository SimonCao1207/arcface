import torch
from einops import rearrange
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

train_df = pd.read_csv("./data/train.csv")
tmp = train_df.groupby("label_group")["posting_id"].agg("unique").to_dict()
train_df["target"] = train_df.label_group.map(tmp)
BASE = "./data/train_images"

class Shopee(Dataset):
    def __init__(self, df, img_size=224, path=''):
        self.df = df
        self.img_size = img_size
        self.path = path
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.df.iloc[idx]["image"])
        image = Image.open(img_path)
        label_group = self.df.iloc[idx]["label_group"]
        return self.transform(image), label_group

if __name__ == "__main__":
    train_data = Shopee(train_df, path=BASE)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    train_features, _ = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    img = train_features[0].squeeze()
    plt.imshow(rearrange(img, 'b w h -> w h b'))
    plt.axis("off")
    plt.show()