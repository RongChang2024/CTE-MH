import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms

from Model import swin_tiny_patch4_window7_224 as create_model
from Dataset import MyDataSet

data_path = ""
csv_path = os.path.join(data_path, "label.csv")
weights_path = ""   

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

label_df = pd.read_csv(csv_path)

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

dataset = MyDataSet(image_dir=data_path, label_df=label_df, transform=data_transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

model = create_model(num_classes=1).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

features = []
ids = []

with torch.no_grad():
    for image_ids, images, labels in tqdm(loader):
        images = images.to(device)
        
        x1 = images.unsqueeze(1)
        x1 = model.conv3d1(x1)
        x1 = model.bn1(x1)
        x1 = model.relu1(x1)
        x1 = torch.squeeze(x1, dim=2)
        x = images + x1
        x, H, W = model.patch_embed(x)
        x = model.pos_drop(x)
        for layer in model.layers:
            x, H, W = layer(x, H, W)
        x = model.norm(x)
        x = model.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        feats = x.cpu().numpy()
        # ===
        
        features.append(feats)
        ids.extend(image_ids)

features = np.concatenate(features, axis=0)

df = pd.DataFrame(features)
df.insert(0, "ID", ids)
df.to_csv("", index=False)
