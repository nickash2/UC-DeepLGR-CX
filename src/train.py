import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deeplgr import DeepLGR
from deeplgr_extended import create_baseline_model, create_extended_model


class TaxiBJDataset(Dataset):
    """PyTorch Dataset for TaxiBJ preprocessed data."""
    
    def __init__(self, data_path):
        """
        Load preprocessed data from npz file.
        
        Args:
            data_path: Path to .npz file with preprocessed data
        """
        print(f"Loading dataset from {data_path}...")
        data = np.load(data_path)
        
        self.X_closeness = torch.FloatTensor(data["X_closeness"])
        self.X_period = torch.FloatTensor(data["X_period"])
        self.X_trend = torch.FloatTensor(data["X_trend"])
        self.X_external = torch.FloatTensor(data["X_external"])
        self.Y = torch.FloatTensor(data["Y"])
        
        print(f"Loaded {len(self.Y)} samples")
        print(f"  X_closeness: {self.X_closeness.shape}")
        print(f"  X_period: {self.X_period.shape}")
        print(f"  X_trend: {self.X_trend.shape}")
        print(f"  X_external: {self.X_external.shape}")
        print(f"  Y: {self.Y.shape}")
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        """
        Return a sample with closeness, period, trend, external features, and target.
        """
        return (
            self.X_closeness[idx],
            self.X_period[idx],
            self.X_trend[idx],
            self.X_external[idx],
            self.Y[idx]
        )


class Trainer:
    """Trainer for DeepLGR models (baseline or extended)."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        use_external=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=0.001,
        checkpoint_dir="checkpoints"
    ):
        """
        Initialize trainer.
        
        Args:
            model: DeepLGR model (baseline or extended)
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            use_external: Whether model uses external features
            device: Device to train on
            lr: Learning rate
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_external = use_external
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Loss and optimizer (as per original DeepLGR and proposal)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
        
        print(f"Trainer initialized on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Using external features: {use_external}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for X_c, X_p, X_t, X_ext, Y in pbar:
            # Move to device
            X_c, X_p, X_t, X_ext, Y = (
                X_c.to(self.device),
                X_p.to(self.device),
                X_t.to(self.device),
                X_ext.to(self.device),
                Y.to(self.device)
            )
            
            # Reshape inputs: [b, t, c, h, w] -> [b, c*t, h, w]
            b = X_c.shape[0]
            X_c_flat = X_c.reshape(b, -1, X_c.shape[-2], X_c.shape[-1])
            X_p_flat = X_p.reshape(b, -1, X_p.shape[-2], X_p.shape[-1])
            X_t_flat = X_t.reshape(b, -1, X_t.shape[-2], X_t.shape[-1])
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_external:
                prediction = self.model((X_c_flat, X_p_flat, X_t_flat), X_ext)
            else:
                prediction = self.model((X_c_flat, X_p_flat, X_t_flat))
            
            loss = self.criterion(prediction, Y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for X_c, X_p, X_t, X_ext, Y in pbar:
                # Move to device
                X_c, X_p, X_t, X_ext, Y = (
                    X_c.to(self.device),
                    X_p.to(self.device),
                    X_t.to(self.device),
                    X_ext.to(self.device),
                    Y.to(self.device)
                )
                
                # Reshape inputs
                b = X_c.shape[0]
                X_c_flat = X_c.reshape(b, -1, X_c.shape[-2], X_c.shape[-1])
                X_p_flat = X_p.reshape(b, -1, X_p.shape[-2], X_p.shape[-1])
                X_t_flat = X_t.reshape(b, -1, X_t.shape[-2], X_t.shape[-1])
                
                # Forward pass
                if self.use_external:
                    prediction = self.model((X_c_flat, X_p_flat, X_t_flat), X_ext)
                else:
                    prediction = self.model((X_c_flat, X_p_flat, X_t_flat))
                
                loss = self.criterion(prediction, Y)
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "history": self.history
        }
        
        # Always save best checkpoint when improvement
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pth")
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
    
    def train(self, n_epochs, patience=10):
        """
        Train model for n_epochs with early stopping.
        
        Args:
            n_epochs: Maximum number of epochs
            patience: Number of epochs without improvement before stopping
        """
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(1, n_epochs + 1):
            print(f"\nEpoch {epoch}/{n_epochs}")
            
            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            # Check improvement
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                patience_counter += 1
            
            # Print summary
            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} | Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping at epoch {epoch}")
                break
        
        # Save final history
        history_path = os.path.join(self.checkpoint_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "="*60)
        print(f"✓ Training complete! Best val loss: {best_val_loss:.6f}")
        print("="*60 + "\n")


def main(args):
    """Main training function."""
    # Set device first
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and args.device == "cuda":
        print("⚠ CUDA requested but not available, falling back to CPU")
    
    print("\n" + "="*60)
    print("DeepLGR Training")
    print("="*60)
    print(f"Model: {'Extended (w/ external)' if args.use_external else 'Baseline'}")
    
    # Parse years to determine file suffix
    if "," in args.year or "-" in args.year:
        # Multi-year format
        if "-" in args.year and "," not in args.year:
            start, end = args.year.split("-")
            year_suffix = f"{start.zfill(2)}-{end.zfill(2)}"
        else:
            years = [y.strip().zfill(2) for y in args.year.split(",")]
            year_suffix = f"{years[0]}-{years[-1]}"
    else:
        # Single year
        year_suffix = args.year.zfill(2)
    
    print(f"Data: {args.data_dir}/BJ{year_suffix}_{{train,val}}.npz")
    print(f"Device: {device}")
    print(f"Batch: {args.batch_size} | LR: {args.lr} | Epochs: {args.epochs}")
    print("="*60)
    
    # Load data
    train_path = os.path.join(args.data_dir, f"BJ{year_suffix}_train.npz")
    val_path = os.path.join(args.data_dir, f"BJ{year_suffix}_val.npz")
    
    train_dataset = TaxiBJDataset(train_path)
    val_dataset = TaxiBJDataset(val_path)
    
    # Use num_workers=0 on CPU to avoid multiprocessing issues
    num_workers = 0 if device == "cpu" else args.num_workers
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )
    
    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    
    # Create model
    t_params = (args.len_closeness, args.len_period, args.len_trend)
    
    if args.use_external:
        model = create_extended_model(t_params=t_params, height=32, width=32, n_external_features=21)
    else:
        model = create_baseline_model(t_params=t_params, height=32, width=32)
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_external=args.use_external,
        device=device,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )
    
    trainer.train(n_epochs=args.epochs, patience=args.patience)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train DeepLGR model (baseline or extended)",
        epilog="""
Examples:
  # Train on single year
  python src/train.py --year 16 --use_external
  
  # Train on multiple years
  python src/train.py --year 13,14,15,16 --use_external
  
  # Train on year range
  python src/train.py --year 13-16 --use_external
        """
    )
    parser.add_argument("--use_external", action="store_true",
                        help="Use extended model with external features (default: baseline)")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Directory with preprocessed data")
    parser.add_argument("--year", type=str, default="16",
                        help="Year(s) to train on: single (16), comma-separated (13,14,15,16), or range (13-16)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--len_closeness", type=int, default=12,
                        help="Number of closeness steps")
    parser.add_argument("--len_period", type=int, default=3,
                        help="Number of period steps")
    parser.add_argument("--len_trend", type=int, default=3,
                        help="Number of trend steps")
    
    args = parser.parse_args()
    main(args)
