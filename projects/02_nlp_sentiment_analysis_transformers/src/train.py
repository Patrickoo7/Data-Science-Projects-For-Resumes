"""
Training script for sentiment analysis with transformers.
Includes advanced features: mixed precision, gradient accumulation, warmup.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import wandb

from model import SentimentTransformer


class TextDataset(Dataset):
    """Dataset for text classification."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class SentimentTrainer:
    """Trainer class for sentiment analysis models."""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device

        # Optimizer with weight decay
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-8
        )

        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * config.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision training
        self.scaler = GradScaler() if config.use_amp else None

        # Metrics
        self.best_val_acc = 0.0

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        predictions, true_labels = [], []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass with mixed precision
            if self.config.use_amp:
                with autocast():
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

            else:
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

            total_loss += loss.item()

            # Collect predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        return avg_loss, accuracy, f1

    def validate(self):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        predictions, true_labels = [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        return avg_loss, accuracy, f1, predictions, true_labels

    def train(self):
        """Main training loop."""
        print("Starting training...")
        print(f"Model: {self.config.model_name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print("-" * 50)

        for epoch in range(self.config.epochs):
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, val_f1, val_preds, val_labels = self.validate()

            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
            print("-" * 50)

            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'train_f1': train_f1,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'epoch': epoch
                })

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, f'{self.config.output_dir}/best_model.pth')
                print(f"✓ Best model saved! Val Acc: {val_acc:.4f}")

        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./models')
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Configuration:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    print("\nNote: Load your dataset and create data loaders")
    print("Training script is ready!")


if __name__ == "__main__":
    main()
