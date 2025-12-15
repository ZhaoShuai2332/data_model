# -*- coding: utf-8 -*-
"""
Transformer Trainer
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc, f1_score
import warnings
warnings.filterwarnings("ignore")

from transformer import TabularTransformer, FocalLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TransformerPredictor:
    """Transformer Trainer."""
    
    def __init__(self, input_dim, d_model=128, num_heads=8, 
                 num_layers=4, d_ff=512, dropout=0.1,
                 learning_rate=0.0001, use_focal_loss=True):
        
        self.model = TabularTransformer(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = FocalLoss() if use_focal_loss else nn.BCELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_f1s = []
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs, _ = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs, _ = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_preds = (all_probs >= 0.5).astype(int)
        
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, roc_auc, f1, all_probs, all_labels
    
    def fit(self, X_train, y_train, X_val, y_val, 
            epochs=50, batch_size=128, patience=15):
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\nStarting Transformer model training...")
        print(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
        
        best_auc = 0
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc, val_f1, _, _ = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            self.val_f1s.append(val_f1)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val AUC: {val_auc:.4f}, "
                      f"Val F1: {val_f1:.4f}")
            
            if val_auc > best_auc or val_f1 > best_f1:
                best_auc = max(best_auc, val_auc)
                best_f1 = max(best_f1, val_f1)
                patience_counter = 0
                torch.save(self.model.state_dict(), 
                          'transformer_model\\outputs\\best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        self.model.load_state_dict(
            torch.load('transformer_model\\outputs\\best_model.pth')
        )
        print(f"\nTraining completed! Best AUC: {best_auc:.4f}, Best F1: {best_f1:.4f}")
        
        return best_auc, best_f1
    
    def predict(self, X):
        self.model.eval()
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for X_batch, in loader:
                X_batch = X_batch.to(device)
                outputs, _ = self.model(X_batch)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)
    
    def get_attention_weights(self, X):
        """Get attention weights for visualization."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X[:1]).to(device)
        
        with torch.no_grad():
            _, attention_weights_list = self.model(X_tensor)
        
        return attention_weights_list
    
    def plot_training_history(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Transformer Loss Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # AUC
        axes[1].plot(self.val_aucs, color='green', label='Val AUC')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Transformer Validation AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # F1
        axes[2].plot(self.val_f1s, color='orange', label='Val F1')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_title('Transformer Validation F1')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('transformer_model\\outputs\\training_history.png', dpi=150)
        print("\nTraining history saved: transformer_model\\outputs\\training_history.png")
        plt.close()
        return fig
