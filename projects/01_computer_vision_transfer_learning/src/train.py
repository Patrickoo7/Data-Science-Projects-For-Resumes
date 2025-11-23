"""
Training script with advanced features:
- Mixed precision training
- Learning rate scheduling
- Early stopping
- MLflow experiment tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np
import mlflow
import mlflow.pytorch

from model import create_model
from data_loader import create_data_loaders
from utils import AverageMeter, accuracy, EarlyStopping


class Trainer:
    """Advanced trainer class with modern deep learning techniques."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        config
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

        # Mixed precision training
        self.scaler = GradScaler() if config.use_amp else None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            verbose=True,
            delta=config.min_delta
        )

        # Metrics tracking
        self.best_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        losses = AverageMeter()
        accs = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}')

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = images.size(0)

            # Mixed precision training
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Calculate accuracy
            acc = accuracy(outputs, labels)

            # Update metrics
            losses.update(loss.item(), batch_size)
            accs.update(acc, batch_size)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accs.avg:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        return losses.avg, accs.avg

    def validate(self):
        """Validate model."""
        self.model.eval()
        losses = AverageMeter()
        accs = AverageMeter()

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                images, labels = images.to(self.device), labels.to(self.device)
                batch_size = images.size(0)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                acc = accuracy(outputs, labels)

                losses.update(loss.item(), batch_size)
                accs.update(acc, batch_size)

        return losses.avg, accs.avg

    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model: {self.config.model_name}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.lr}")
        print("-" * 50)

        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            epoch_time = time.time() - start_time

            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            }, step=epoch)

            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Time: {epoch_time:.2f}s")
            print("-" * 50)

            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                save_path = Path(self.config.save_dir) / f'best_model_{self.config.model_name}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, save_path)
                print(f"✓ Best model saved! Accuracy: {val_acc:.2f}%")

            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        print(f"\nTraining completed! Best validation accuracy: {self.best_acc:.2f}%")
        return self.best_acc


def main():
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument('--model', type=str, default='resnet50', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum improvement')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = create_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=True
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Note: Data loaders should be created from actual dataset
    # This is a placeholder - implement based on your dataset
    print("Note: Implement data loaders based on your dataset")

    print("Training configuration ready!")


if __name__ == "__main__":
    main()
