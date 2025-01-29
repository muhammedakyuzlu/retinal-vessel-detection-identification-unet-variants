import time
from glob import glob

import torch
from torch.utils.data import DataLoader

from data import DriveDataset
from models.unet_model import UNet
from losses.unet_loss import DiceBCELoss
from losses import seeding, create_dir, epoch_time

class UNetTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda')
        self.model = UNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['LEARNING_RATE'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.config['PATIENCE'], verbose=True)
        self.loss_fn = DiceBCELoss()

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(self.device, dtype=torch.float32), y.to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            print("loss: " + str(total_loss))
        return total_loss / len(loader)

    def evaluate_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device, dtype=torch.float32), y.to(self.device, dtype=torch.float32)
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                total_loss += loss.item()
        return total_loss / len(loader)

    def train(self, train_loader, valid_loader):
        best_valid_loss = float("inf")
        no_improve_count = 0

        for epoch in range(self.config['EPOCHS']):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            valid_loss = self.evaluate_epoch(valid_loader)
            self.scheduler.step(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.config['CHECKPOINT_PATH']+str(time.time())+".pth")
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > self.config['PATIENCE']:
                    print("Early stopping triggered.")
                    break

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

if __name__ == "__main__":
    seeding(42)
    # create_dir("files")

    CONFIG = {
        'SIZE': (512, 512),
        'BATCH_SIZE': 2,
        'EPOCHS': 150,
        'LEARNING_RATE': 1e-3,
        'CHECKPOINT_PATH': "/checkpoints/",
        'PATIENCE': 100,
        'TRAIN_IMG_PATH':  "/train/images/*",
        'TRAIN_MASK_PATH': "/train/masks/*",
        'VALID_IMG_PATH':  "/test/images/*",
        'VALID_MASK_PATH': "/test/masks/*"
    }

    train_dataset = DriveDataset(sorted(glob(CONFIG['TRAIN_IMG_PATH'])), sorted(glob(CONFIG['TRAIN_MASK_PATH'])))
    valid_dataset = DriveDataset(sorted(glob(CONFIG['VALID_IMG_PATH'])), sorted(glob(CONFIG['VALID_MASK_PATH'])))
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2)

    trainer = UNetTrainer(CONFIG)
    trainer.train(train_loader, valid_loader)
