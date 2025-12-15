import torch
import torch.nn as nn
from res_net import ResNet1D, FocalLoss
from evaluate import MetricsCalculator
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np


# Use GPU for torch network model acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class ResNetPredictor:
    """ResNet trainer with F1-score"""
    
    def __init__(self, input_dim, layers=[2, 2, 2, 2], 
                 channels=[64, 128, 256, 512],
                 dropout_rate=0.3, learning_rate=0.001, 
                 use_focal_loss=True):
        
        self.model = ResNet1D(
            input_dim=input_dim,
            layers=layers,
            channels=channels,
            dropout_rate=dropout_rate
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = FocalLoss() if use_focal_loss else nn.BCELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_f1s = []
        self.metrics_calculator = MetricsCalculator()
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = self.model(X_batch).squeeze()
            loss = self.criterion(outputs, y_batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, threshold=0.5):
        self.model.eval()
        total_loss = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = self.model(X_batch).squeeze()
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_preds = (all_probs >= threshold).astype(int)
        
        # Calculate all metrics
        metrics = self.metrics_calculator.calculate_metrics(
            all_labels, all_preds, all_probs, threshold
        )
        
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, metrics, all_probs, all_labels
    
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
        
        print(f"\nStart training ResNet1D model...")
        print(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
        
        best_f1 = 0
        best_auc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, metrics, _, _ = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(metrics['roc_auc'])
            self.val_f1s.append(metrics['f1_score'])
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val AUC: {metrics['roc_auc']:.4f}, "
                      f"Val F1: {metrics['f1_score']:.4f}")
            
            # Consider both AUC and F1-score
            if metrics['f1_score'] > best_f1 or metrics['roc_auc'] > best_auc:
                best_f1 = max(best_f1, metrics['f1_score'])
                best_auc = max(best_auc, metrics['roc_auc'])
                patience_counter = 0
                torch.save(self.model.state_dict(), 'cnn_model\\outputs\\best_resnet1d_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        self.model.load_state_dict(torch.load('cnn_model\\outputs\\best_resnet1d_model.pth'))
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
                outputs = self.model(X_batch).squeeze()
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)
    
    def plot_training_history(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss Curve
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('ResNet1D Loss Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # AUC Curve
        axes[1].plot(self.val_aucs, color='green', label='Val AUC')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('ResNet1D Validation AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # F1 Curve
        axes[2].plot(self.val_f1s, color='orange', label='Val F1-Score')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_title('ResNet1D Validation F1-Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cnn_model\\outputs\\training_history.png', dpi=150)
        print("\nTraining history saved: cnn_model\\outputs\\training_history.png")
        return fig
