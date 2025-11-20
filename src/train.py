import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deeplgr_extended import create_baseline_model, create_extended_model


class TaxiBJDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.X_closeness = torch.FloatTensor(data["X_closeness"])
        self.X_period = torch.FloatTensor(data["X_period"])
        self.X_trend = torch.FloatTensor(data["X_trend"])
        self.X_external = torch.FloatTensor(data["X_external"])
        self.Y = torch.FloatTensor(data["Y"])

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (
            self.X_closeness[idx],
            self.X_period[idx],
            self.X_trend[idx],
            self.X_external[idx],
            self.Y[idx],
        )


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        use_external=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=1e-4,
        checkpoint_dir="checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_external = use_external
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for X_c, X_p, X_t, X_ext, Y in pbar:
            X_c, X_p, X_t, X_ext, Y = (
                X_c.to(self.device),
                X_p.to(self.device),
                X_t.to(self.device),
                X_ext.to(self.device),
                Y.to(self.device),
            )

            b = X_c.shape[0]
            X_c_flat = X_c.reshape(b, -1, X_c.shape[-2], X_c.shape[-1])
            X_p_flat = X_p.reshape(b, -1, X_p.shape[-2], X_p.shape[-1])
            X_t_flat = X_t.reshape(b, -1, X_t.shape[-2], X_t.shape[-1])

            self.optimizer.zero_grad()

            if self.use_external:
                prediction = self.model((X_c_flat, X_p_flat, X_t_flat), X_ext)
            else:
                prediction = self.model((X_c_flat, X_p_flat, X_t_flat))

            loss = self.criterion(prediction, Y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for X_c, X_p, X_t, X_ext, Y in pbar:
                X_c, X_p, X_t, X_ext, Y = (
                    X_c.to(self.device),
                    X_p.to(self.device),
                    X_t.to(self.device),
                    X_ext.to(self.device),
                    Y.to(self.device),
                )

                b = X_c.shape[0]
                X_c_flat = X_c.reshape(b, -1, X_c.shape[-2], X_c.shape[-1])
                X_p_flat = X_p.reshape(b, -1, X_p.shape[-2], X_p.shape[-1])
                X_t_flat = X_t.reshape(b, -1, X_t.shape[-2], X_t.shape[-1])

                if self.use_external:
                    prediction = self.model((X_c_flat, X_p_flat, X_t_flat), X_ext)
                else:
                    prediction = self.model((X_c_flat, X_p_flat, X_t_flat))

                loss = self.criterion(prediction, Y)
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "history": self.history,
        }

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best.pth")
            torch.save(checkpoint, best_path)

    def train(self, n_epochs, patience=10):
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, n_epochs + 1):
            print(f"Epoch {epoch}/{n_epochs}")

            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                patience_counter += 1

            print(
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Best: {best_val_loss:.6f} | Patience: {patience_counter}/{patience}"
            )

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            self.scheduler.step(val_loss)

        history_path = os.path.join(self.checkpoint_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"Training complete. Best: {best_val_loss:.6f}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model: {'Extended' if args.use_external else 'Baseline'}")
    print(f"Device: {device} | Batch: {args.batch_size} | LR: {args.lr}")

    train_path = os.path.join(args.data_dir, f"BJ{args.year}_train.npz")
    val_path = os.path.join(args.data_dir, f"BJ{args.year}_val.npz")

    train_dataset = TaxiBJDataset(train_path)
    val_dataset = TaxiBJDataset(val_path)

    num_workers = 0 if device == "cpu" else args.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    t_params = (args.len_closeness, args.len_period, args.len_trend)

    if args.use_external:
        model = create_extended_model(
            t_params=t_params, height=32, width=32, n_external_features=21
        )
    else:
        model = create_baseline_model(t_params=t_params, height=32, width=32)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_external=args.use_external,
        device=device,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer.train(n_epochs=args.epochs, patience=args.patience)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepLGR")
    parser.add_argument("--use_external", action="store_true")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--year", type=str, default="16")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--len_closeness", type=int, default=12)
    parser.add_argument("--len_period", type=int, default=3)
    parser.add_argument("--len_trend", type=int, default=3)

    args = parser.parse_args()
    main(args)
