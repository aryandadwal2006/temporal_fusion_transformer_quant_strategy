import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import gc
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime

class EnhancedTFTTrainer:
    """
    OPTIMIZED: Enhanced trainer with better memory management
    Trains model to predict price slope direction (3-class classification)
    """
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model = self.model.to(device)
        
        # Optimizer with gradient clipping
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Focal loss for handling class imbalance
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.LR_FACTOR,
            patience=config.LR_PATIENCE,
            min_lr=config.LR_MIN,
            #verbose=True
        )
        
        # Mixed precision training
        self.use_amp = config.USE_GPU
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        self.model_saved = False
        
        print("="*60)
        print("TRAINER INITIALIZED")
        print("="*60)
        print(f"Device: {device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Initial LR: {config.LEARNING_RATE}")
        print(f"Prediction: {config.SLOPE_PREDICTION_WINDOW}s price slope")
        print("="*60 + "\n")
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch with memory-efficient batch processing
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        nan_batches = 0
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
            
            # Check for NaN loss
            if torch.isnan(loss):
                nan_batches += 1
                continue
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRADIENT_CLIP_VAL
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRADIENT_CLIP_VAL
                )
                self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Logging
            if (batch_idx + 1) % self.config.LOG_INTERVAL == 0:
                avg_loss = total_loss / (batch_idx + 1 - nan_batches)
                acc = 100 * correct / total
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
            
            # Periodic cleanup for long training runs
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if nan_batches > 0:
            print(f"  ⚠️  WARNING: {nan_batches} batches had NaN values")
        
        avg_loss = total_loss / max(len(train_loader) - nan_batches, 1)
        accuracy = 100 * correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """
        Validate the model with detailed class-wise metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        valid_batches = 0
        
        # Track class-wise accuracy
        class_correct = [0, 0, 0]
        class_total = [0, 0, 0]
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(sequences)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(sequences)
                    loss = self.criterion(outputs, targets)
                
                # Skip NaN losses
                if torch.isnan(loss):
                    continue
                
                total_loss += loss.item()
                valid_batches += 1
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Class-wise accuracy
                for i in range(3):
                    mask = (targets == i)
                    class_total[i] += mask.sum().item()
                    class_correct[i] += ((predicted == targets) & mask).sum().item()
        
        avg_loss = total_loss / max(valid_batches, 1)
        accuracy = 100 * correct / total if total > 0 else 0
        
        # Calculate class-wise accuracy
        class_acc = [
            100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            for i in range(3)
        ]
        
        # Memory cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return avg_loss, accuracy, class_acc
    
    def train(self, train_loader, val_loader, epochs: int):
        """
        Main training loop with early stopping and checkpointing
        """
        print("="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Max epochs: {epochs}")
        print(f"Early stopping patience: {self.config.EARLY_STOPPING_PATIENCE}")
        print("="*60 + "\n")
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, class_acc = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  Class Accuracy:")
            print(f"    Down (0):    {class_acc[0]:.1f}%")
            print(f"    Neutral (1): {class_acc[1]:.1f}%")
            print(f"    Up (2):      {class_acc[2]:.1f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rate'].append(current_lr)
            
            # Check for NaN
            if np.isnan(train_loss) or np.isnan(val_loss):
                print("\n❌ ERROR: Training producing NaN! Stopping...")
                break
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print("  ✓ Best model saved!")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.SAVE_CHECKPOINT_EVERY == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Stop if learning rate too small
            if current_lr < self.config.LR_MIN:
                print("\n⚠️  Learning rate reached minimum. Stopping training.")
                break
            
            print()
            
            # Memory cleanup between epochs
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Save final checkpoint if not saved yet
        if not self.model_saved:
            self.save_checkpoint(epochs-1, is_best=False)
        
        # Save training history
        self.save_training_history()
        
        print("="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Val Acc:  {self.best_val_acc:.2f}%")
        print("="*60 + "\n")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'training_history': self.training_history
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if is_best:
            path = f"{self.config.MODEL_DIR}/best_tft_model.pt"
            torch.save(checkpoint, path)
            self.model_saved = True
        else:
            path = f"{self.config.MODEL_DIR}/checkpoint_epoch_{epoch+1}.pt"
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.training_history = checkpoint['training_history']
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✓ Checkpoint loaded from {path}")
    
    def save_training_history(self):
        """
        Save training history to JSON
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f"{self.config.RESULTS_DIR}/training_history_{timestamp}.json"
        
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=4)
        
        print(f"✓ Training history saved to {path}")


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses training on hard examples
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def test_trainer():
    """Test trainer initialization"""
    from config import Config
    from tft_model_enhanced import TemporalFusionTransformer
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TemporalFusionTransformer(
        num_features=config.get_num_features(),
        hidden_size=config.HIDDEN_SIZE,
        lstm_layers=config.LSTM_LAYERS,
        num_attention_heads=config.ATTENTION_HEADS,
        dropout=config.DROPOUT,
        ffn_hidden_size=config.FFN_HIDDEN_SIZE
    )
    
    trainer = EnhancedTFTTrainer(model, config, device)
    print("✓ Trainer test successful!")

if __name__ == "__main__":
    test_trainer()