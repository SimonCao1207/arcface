import timm
import torch.optim as optim
from tqdm import tqdm
from data import *
from loss import *

def convert_to_one_hot(labels):
    # Convert to one-hot encoding
    label_indices = torch.tensor([torch.where(label_groups == label)[0].item() for label in labels])
    one_hot_labels = torch.zeros(len(labels), num_classes)
    one_hot_labels.scatter_(1, label_indices.unsqueeze(1), 1)
    return one_hot_labels

train_df = pd.read_csv("./data/train.csv")
tmp = train_df.groupby("label_group")["posting_id"].agg("unique").to_dict()
train_df["target"] = train_df.label_group.map(tmp)
label_groups = train_df.label_group.unique()
num_classes = len(label_groups)

BASE = "./data/train_images"
model_name = "resnet50d.ra4_e3600_r224_in1k"
model = timm.create_model(model_name, pretrained=True, num_classes=0)
train_data = Shopee(train_df, path=BASE)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
train_features, labels = next(iter(train_loader))
one_hot_labels = convert_to_one_hot(labels)
embeddings = F.normalize(model(train_features), dim=1, p=2)

arcface_loss = ArcLoss(2048, num_classes)
loss = arcface_loss(embeddings, one_hot_labels)
print(loss)