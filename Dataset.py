from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage import io
from skimage import io, exposure, img_as_uint, img_as_float
import torchvision
import SimpleITK
import os

class MyDataSet(Dataset):
    def __init__(self, image_dir, label_df, transform=None):
        self.image_dir = image_dir
        self.label_df = label_df
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        image_id = self.label_df.iloc[idx, 0]
        label = int(self.label_df.iloc[idx, 1])

        img_path = os.path.join(self.image_dir, image_id)
        image = np.load(img_path).astype(np.float32)  # shape: (3, H, W)

        if self.transform:
            image = self.transform(image)  # 不要 torch.from_numpy(image)
        else:
            image = torch.from_numpy(image)
        
        return image_id, image, label

    @staticmethod
    def collate_fn(batch):
        image_ids, images, labels = zip(*batch)
        return list(image_ids), torch.stack(images), torch.tensor(labels)
