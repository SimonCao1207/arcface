import torch
from einops import rearrange
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.io import read_image
import matplotlib.pyplot as plt
from PIL import Image

train_df = pd.read_csv("./data/train.csv")
tmp = train_df.groupby("label_group")["posting_id"].agg("unique").to_dict()
train_df["target"] = train_df.label_group.map(tmp)
BASE = "./data/train_images"

class Shopee(Dataset):
    def __init__(self, df, img_size=256, path=''):
        self.df = df
        self.img_size = img_size
        self.path = path
        self.transform = torchvision.transforms.Resize(img_size)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.df.iloc[idx]["image"])
        image = read_image(img_path)
        return self.transform(image)

if __name__ == "__main__":
    train_data = Shopee(train_df, path=BASE)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    train_features = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    img = train_features[0].squeeze()
    plt.imshow(rearrange(img, 'b w h -> w h b'))
    plt.axis("off")
    plt.show()