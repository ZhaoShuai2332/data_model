import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")
from mlp import RiskPredictionNet, FocalLoss

# Use GPU to accelerate torch network training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

"""
Code for model training and prediction
"""
class RiskPredictor:
    """Model training and prediction class"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], 
                 dropout_rate=0.3, learning_rate=0.001, 
                 use_focal_loss=True):
        
        self.model = RiskPredictionNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Use Focal Loss to handle unbalanced data
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.criterion = nn.BCELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            outputs = self.model(X_batch).squeeze()
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = self.model(X_batch).squeeze()
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        # Calculate AUC
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, roc_auc, np.array(all_preds), np.array(all_labels)
    
    def fit(self, X_train, y_train, X_val, y_val, 
            epochs=50, batch_size=256, patience=10):
        """Train model"""
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        print(f"\nStarting model training...")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        best_auc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_auc, _, _ = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val AUC: {val_auc:.4f}")
            
            # Early Stopping
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'mlp_model\\outputs\\best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('mlp_model\\outputs\\best_model.pth'))
        print(f"\nTraining completed! Best validation AUC: {best_auc:.4f}")
        
        return best_auc
    
    def predict(self, X):
        """Predict probabilities"""
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
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        axes[0].plot(self.train_losses, label='Train Loss', alpha=0.8)
        axes[0].plot(self.val_losses, label='Val Loss', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # AUC curve
        axes[1].plot(self.val_aucs, label='Val AUC', color='green', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Validation AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mlp_model\\outputs\\training_history.png', dpi=150, bbox_inches='tight')
        print("\nTraining history plot saved to:  mlp_model\\outputs\\training_history.png")
        
        return fig
