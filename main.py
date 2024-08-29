import sys
import timm
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from data import *
from loss import *

train_df = pd.read_csv("./data/train.csv")
tmp = train_df.groupby("label_group")["posting_id"].agg("unique").to_dict()
train_df["target"] = train_df.label_group.map(tmp)
label_groups = torch.tensor(train_df.label_group.unique())
num_classes = len(label_groups)

def convert_to_one_hot(labels):
    # Convert to one-hot encoding
    label_indices = torch.tensor([torch.where(label_groups == label)[0].item() for label in labels])
    one_hot_labels = torch.zeros(len(labels), num_classes)
    one_hot_labels.scatter_(1, label_indices.unsqueeze(1), 1)
    return one_hot_labels

BASE = "./data/train_images"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.01
in_features = 2048
out_features = num_classes 
num_epochs = 1

model_name = "resnet50d.ra4_e3600_r224_in1k"
model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay = 0.0005)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 25, eta_min=lr*0.01)
arcface_loss = ArcLoss(in_features, out_features)

train_data = Shopee(train_df, path=BASE)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = []
    for step, (train_features, labels) in tqdm(enumerate(train_loader), desc= 'Training', leave=False, total=len(train_loader)):
        out = model(train_features.to(device))
        embeddings = F.normalize(out, dim=1, p=2)
        one_hot_labels = convert_to_one_hot(labels).to(device)
        loss = arcface_loss(embeddings, one_hot_labels)
        optimizer.zero_grad()  
        loss.backward()    
        optimizer.step()
        loss_item = loss.cpu().detach().numpy()
        running_loss.append(loss_item) 
        print("Epoch: {}/{} - Learning rate: {:.6f} - Min/Avg/Max train Loss: {:.6f}/{:.6f}/{:.6f}".format(epoch+1, num_epochs, optimizer.state_dict()['param_groups'][0]['lr'], min(running_loss), np.mean(running_loss), max(running_loss)))
        scheduler.step(epoch + step / len(train_loader))
    mean_loss = np.mean(running_loss)
    model.eval()