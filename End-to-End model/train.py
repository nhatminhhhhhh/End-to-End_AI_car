import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

from config import DATA_DIR, MODEL_DIR, BATCH_SIZE, LEARNING_RATE, EPOCHS, PATIENCE, VAL_SPLIT
from model import E2EModel
from dataset import E2EDataset


def train():
    parser = argparse.ArgumentParser(description='Train E2E driving model')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--save-dir', type=str, default=MODEL_DIR)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = E2EDataset(args.data_dir, augment=True)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"[Train] Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = E2EModel().to(device)
    # Adam optimizer điều chỉnh tốc độ học tự động
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # SGD optimizer sử dụng momentum
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # Mean Squared Error loss function
    criterion = nn.MSELoss()
    # ReduceLROnPlateau điều chỉnh tốc độ học
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() # reset gradient
            outputs = model(images)
            loss = criterion(outputs, labels) # tính loss
            loss.backward() # backpropagation
            optimizer.step() # update weights
            train_loss += loss.item() * images.size(0) # tính train_loss mỗi epoch
        train_loss /= len(train_ds) # tính train_loss trung bình

        model.eval() # đánh giá mô hình
        val_loss = 0.0
        with torch.no_grad(): # tắt gradient
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_ds) # tính val_loss trung bình

        scheduler.step(val_loss) # điều chỉnh tốc độ học

        print(f"Epoch {epoch:3d}/{args.epochs}  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved {model_path}")

            dummy = torch.randn(1, 3, 160, 120).to(device) # test model
            traced = torch.jit.trace(model, dummy) # trace model
            traced_path = os.path.join(args.save_dir, 'best_model_traced.pt') # save model
            traced.save(traced_path) # save model
            print(f"  -> Saved {traced_path}") # print model
        else:
            patience_counter += 1 # tăng patience
            if patience_counter >= PATIENCE: # nếu patience đủ 
                print(f"[Early stopping] No improvement for {PATIENCE} epochs")
                break # dừng huấn luyện

    print(f"\n[Done] Best val loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    train()
