import os
import argparse
import numpy as np
import pandas as pd
import torch
import warnings
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from Dataset import MyDataSet
from Model import swin_tiny_patch4_window7_224 as create_model
from utils_II_CRC import train_one_epoch, evaluate

warnings.filterwarnings("ignore")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    # 读取CSV标签文件
    label_df = pd.read_csv(os.path.join(args.data_path, "labels.csv"))
    train_df, val_df = train_test_split(
        label_df,
        test_size=0.3,
        stratify=label_df['label'],
        random_state=args.seed
    )

    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),              
            transforms.RandomRotation(degrees=10),                
            transforms.ColorJitter(brightness=0.1, contrast=0.1), 
            transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.97, 1.03)), 
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
    }

    train_dataset = MyDataSet(image_dir=args.data_path, label_df=train_df, transform=data_transform["train"])
    val_dataset   = MyDataSet(image_dir=args.data_path, label_df=val_df,   transform=data_transform["val"])

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 16])
    print(f'Using {nw} dataloader workers per process')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=nw, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=nw, collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=1).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not found."
        weights_dict = torch.load(args.weights, map_location=device)
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_auc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_auc = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, scheduler)
        val_loss, val_auc     = evaluate(model, criterion, val_loader, device, epoch, "val")

        tags = ["train_loss", "train_auc", "val_loss", "val_auc", "lr"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_auc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_auc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f"./weights/best_model_epoch{epoch}_auc{val_auc:.3f}.pth")
            print(f"Saved new best model at epoch {epoch} with val AUC {val_auc:.4f}, train AUC {train_auc:.4f}")

    return train_loss, train_auc, val_loss, val_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    main(args)