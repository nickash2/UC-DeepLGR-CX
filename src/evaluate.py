import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import json
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deeplgr import DeepLGR
from deeplgr_extended import create_baseline_model, create_extended_model
from train import TaxiBJDataset


def denormalize_flow(flow_data, stats):
    """
    Denormalize flow data using saved statistics.
    
    Args:
        flow_data: Normalized flow data
        stats: Dict with 'flow_min' and 'flow_max'
        
    Returns:
        Denormalized flow data
    """
    flow_min = stats["flow_min"]
    flow_max = stats["flow_max"]
    return flow_data * (flow_max - flow_min) + flow_min


def compute_mae(predictions, targets):
    """
    Compute Mean Absolute Error as per proposal formula:
    L_MAE = (1/z) * Σ|y_i - ŷ_i|
    
    Args:
        predictions: Predicted values (ŷ)
        targets: Ground truth values (y)
        
    Returns:
        MAE value
    """
    z = predictions.size  # Total number of elements
    mae = (1.0 / z) * np.sum(np.abs(targets - predictions))
    return mae


def compute_smape(predictions, targets):
    """
    Compute Symmetric Mean Absolute Percentage Error as per proposal formula:
    L_SMAPE = (1/z) * Σ(|y_i - ŷ_i|)/(|y_i| + |ŷ_i|)
    
    Args:
        predictions: Predicted values (ŷ)
        targets: Ground truth values (y)
        
    Returns:
        SMAPE value (0-1 scale, multiply by 100 for percentage)
    """
    z = predictions.size
    numerator = np.abs(targets - predictions)
    denominator = np.abs(targets) + np.abs(predictions)
    
    # Handle division by zero: when both y and y hat are 0, contribution is 0
    mask = denominator > 0
    smape_values = np.zeros_like(numerator)
    smape_values[mask] = numerator[mask] / denominator[mask]
    
    smape = (1.0 / z) * np.sum(smape_values)
    return smape * 100  # Return as percentage


class Evaluator:
    """Evaluator for DeepLGR models (baseline or extended)."""
    
    def __init__(
        self,
        model,
        test_loader,
        stats,
        use_external=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir="results"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained DeepLGR model (baseline or extended)
            test_loader: DataLoader for test data
            stats: Normalization statistics
            use_external: Whether model uses external features
            device: Device to evaluate on
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.stats = stats
        self.use_external = use_external
        self.device = device
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Evaluator initialized on device: {device}")
        print(f"Using external features: {use_external}")
    
    def predict_all(self):
        """
        Run inference on entire test set.
        
        Returns:
            predictions: Array of predictions [n_samples, 2, 32, 32]
            targets: Array of ground truth [n_samples, 2, 32, 32]
            external_features: Array of external features [n_samples, 21]
        """
        print("Running inference on test set...")
        
        all_predictions = []
        all_targets = []
        all_external = []
        
        with torch.no_grad():
            for batch_idx, (X_c, X_p, X_t, X_ext, Y) in enumerate(self.test_loader):
                # Move to device
                X_c = X_c.to(self.device)
                X_p = X_p.to(self.device)
                X_t = X_t.to(self.device)
                X_ext = X_ext.to(self.device)
                Y = Y.to(self.device)
                
                # Reshape inputs
                b = X_c.shape[0]
                X_c_flat = X_c.reshape(b, -1, X_c.shape[-2], X_c.shape[-1])
                X_p_flat = X_p.reshape(b, -1, X_p.shape[-2], X_p.shape[-1])
                X_t_flat = X_t.reshape(b, -1, X_t.shape[-2], X_t.shape[-1])
                
                # Predict
                if self.use_external:
                    prediction = self.model((X_c_flat, X_p_flat, X_t_flat), X_ext)
                else:
                    prediction = self.model((X_c_flat, X_p_flat, X_t_flat))
                
                # Store results
                all_predictions.append(prediction.cpu().numpy())
                all_targets.append(Y.cpu().numpy())
                all_external.append(X_ext.cpu().numpy())
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(self.test_loader)} batches")
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        external_features = np.concatenate(all_external, axis=0)
        
        print(f"Inference complete: {len(predictions)} samples")
        return predictions, targets, external_features
    
    def compute_overall_metrics(self, predictions, targets):
        """
        Compute overall MAE and SMAPE metrics.
        
        Args:
            predictions: Predicted flow data (normalized)
            targets: Ground truth flow data (normalized)
            
        Returns:
            Dict with metric values
        """
        print("\nComputing overall metrics...")
        
        # Denormalize for true-scale metrics
        pred_denorm = denormalize_flow(predictions, self.stats)
        target_denorm = denormalize_flow(targets, self.stats)
        
        mae = compute_mae(pred_denorm, target_denorm)
        smape = compute_smape(pred_denorm, target_denorm)
        
        # Also compute for inflow and outflow separately
        mae_inflow = compute_mae(pred_denorm[:, 0], target_denorm[:, 0])
        mae_outflow = compute_mae(pred_denorm[:, 1], target_denorm[:, 1])
        
        smape_inflow = compute_smape(pred_denorm[:, 0], target_denorm[:, 0])
        smape_outflow = compute_smape(pred_denorm[:, 1], target_denorm[:, 1])
        
        metrics = {
            "mae_overall": float(mae),
            "smape_overall": float(smape),
            "mae_inflow": float(mae_inflow),
            "mae_outflow": float(mae_outflow),
            "smape_inflow": float(smape_inflow),
            "smape_outflow": float(smape_outflow)
        }
        
        print(f"  MAE (overall): {mae:.4f}")
        print(f"  SMAPE (overall): {smape:.2f}%")
        print(f"  MAE (inflow): {mae_inflow:.4f}")
        print(f"  MAE (outflow): {mae_outflow:.4f}")
        
        return metrics
    
    def compute_subgroup_metrics(self, predictions, targets, external_features):
        """
        Compute metrics for different subgroups (weather, calendar).
        
        Args:
            predictions: Predicted flow data
            targets: Ground truth flow data
            external_features: External feature array [n_samples, 21]
                Features: [temp, wind, weather(17), weekend, holiday]
        
        Returns:
            Dict with subgroup metrics
        """
        print("\nComputing subgroup metrics...")
        
        # Denormalize
        pred_denorm = denormalize_flow(predictions, self.stats)
        target_denorm = denormalize_flow(targets, self.stats)
        
        subgroup_metrics = {}
        
        # Weekend vs Weekday
        is_weekend = external_features[:, 19] > 0.5  # Index 19 is weekend flag
        is_weekday = ~is_weekend
        
        if np.sum(is_weekend) > 0:
            mae_weekend = compute_mae(pred_denorm[is_weekend], target_denorm[is_weekend])
            smape_weekend = compute_smape(pred_denorm[is_weekend], target_denorm[is_weekend])
            subgroup_metrics["weekend"] = {
                "count": int(np.sum(is_weekend)),
                "mae": float(mae_weekend),
                "smape": float(smape_weekend)
            }
            print(f"  Weekend ({np.sum(is_weekend)} samples): MAE={mae_weekend:.4f}, SMAPE={smape_weekend:.2f}%")
        
        if np.sum(is_weekday) > 0:
            mae_weekday = compute_mae(pred_denorm[is_weekday], target_denorm[is_weekday])
            smape_weekday = compute_smape(pred_denorm[is_weekday], target_denorm[is_weekday])
            subgroup_metrics["weekday"] = {
                "count": int(np.sum(is_weekday)),
                "mae": float(mae_weekday),
                "smape": float(smape_weekday)
            }
            print(f"  Weekday ({np.sum(is_weekday)} samples): MAE={mae_weekday:.4f}, SMAPE={smape_weekday:.2f}%")
        
        # Holiday vs Non-holiday
        is_holiday = external_features[:, 20] > 0.5  # Index 20 is holiday flag
        is_regular = ~is_holiday
        
        if np.sum(is_holiday) > 0:
            mae_holiday = compute_mae(pred_denorm[is_holiday], target_denorm[is_holiday])
            smape_holiday = compute_smape(pred_denorm[is_holiday], target_denorm[is_holiday])
            subgroup_metrics["holiday"] = {
                "count": int(np.sum(is_holiday)),
                "mae": float(mae_holiday),
                "smape": float(smape_holiday)
            }
            print(f"  Holiday ({np.sum(is_holiday)} samples): MAE={mae_holiday:.4f}, SMAPE={smape_holiday:.2f}%")
        
        if np.sum(is_regular) > 0:
            mae_regular = compute_mae(pred_denorm[is_regular], target_denorm[is_regular])
            smape_regular = compute_smape(pred_denorm[is_regular], target_denorm[is_regular])
            subgroup_metrics["regular"] = {
                "count": int(np.sum(is_regular)),
                "mae": float(mae_regular),
                "smape": float(smape_regular)
            }
            print(f"  Regular ({np.sum(is_regular)} samples): MAE={mae_regular:.4f}, SMAPE={smape_regular:.2f}%")
        
        # Weather conditions (simplified: rainy vs non-rainy)
        # Weather indices 3-9 are rain-related (Rainy, Sprinkle, ModerateRain, HeavyRain, etc.)
        weather_one_hot = external_features[:, 2:19]  # Shape: [n_samples, 17]
        rainy_indices = [3, 4, 5, 6, 7, 8, 9]  # Rain-related weather types
        is_rainy = weather_one_hot[:, rainy_indices].sum(axis=1) > 0.5
        is_sunny = ~is_rainy
        
        if np.sum(is_rainy) > 0:
            mae_rainy = compute_mae(pred_denorm[is_rainy], target_denorm[is_rainy])
            smape_rainy = compute_smape(pred_denorm[is_rainy], target_denorm[is_rainy])
            subgroup_metrics["rainy"] = {
                "count": int(np.sum(is_rainy)),
                "mae": float(mae_rainy),
                "smape": float(smape_rainy)
            }
            print(f"  Rainy ({np.sum(is_rainy)} samples): MAE={mae_rainy:.4f}, SMAPE={smape_rainy:.2f}%")
        
        if np.sum(is_sunny) > 0:
            mae_sunny = compute_mae(pred_denorm[is_sunny], target_denorm[is_sunny])
            smape_sunny = compute_smape(pred_denorm[is_sunny], target_denorm[is_sunny])
            subgroup_metrics["sunny"] = {
                "count": int(np.sum(is_sunny)),
                "mae": float(mae_sunny),
                "smape": float(smape_sunny)
            }
            print(f"  Non-rainy ({np.sum(is_sunny)} samples): MAE={mae_sunny:.4f}, SMAPE={smape_sunny:.2f}%")
        
        return subgroup_metrics
    
    def visualize_predictions(self, predictions, targets, n_samples=5):
        """
        Visualize sample predictions vs ground truth.
        
        Args:
            predictions: Predicted flow data
            targets: Ground truth flow data
            n_samples: Number of samples to visualize
        """
        print(f"\nCreating visualizations for {n_samples} samples...")
        
        # Denormalize
        pred_denorm = denormalize_flow(predictions, self.stats)
        target_denorm = denormalize_flow(targets, self.stats)
        
        # Select random samples
        indices = np.random.choice(len(predictions), n_samples, replace=False)
        
        for idx in indices:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Inflow - Ground Truth
            im0 = axes[0, 0].imshow(target_denorm[idx, 0], cmap="hot", interpolation="nearest")
            axes[0, 0].set_title("Inflow - Ground Truth")
            axes[0, 0].axis("off")
            plt.colorbar(im0, ax=axes[0, 0])
            
            # Inflow - Prediction
            im1 = axes[0, 1].imshow(pred_denorm[idx, 0], cmap="hot", interpolation="nearest")
            axes[0, 1].set_title("Inflow - Prediction")
            axes[0, 1].axis("off")
            plt.colorbar(im1, ax=axes[0, 1])
            
            # Outflow - Ground Truth
            im2 = axes[1, 0].imshow(target_denorm[idx, 1], cmap="hot", interpolation="nearest")
            axes[1, 0].set_title("Outflow - Ground Truth")
            axes[1, 0].axis("off")
            plt.colorbar(im2, ax=axes[1, 0])
            
            # Outflow - Prediction
            im3 = axes[1, 1].imshow(pred_denorm[idx, 1], cmap="hot", interpolation="nearest")
            axes[1, 1].set_title("Outflow - Prediction")
            axes[1, 1].axis("off")
            plt.colorbar(im3, ax=axes[1, 1])
            
            mae_sample = compute_mae(pred_denorm[idx], target_denorm[idx])
            plt.suptitle(f"Sample {idx} - MAE: {mae_sample:.4f}")
            
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(self.output_dir, f"sample_{idx}_visualization.png")
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
        
        print(f"Visualizations saved to {self.output_dir}")
    
    def evaluate(self, visualize=True, n_viz_samples=5):
        """
        Run full evaluation pipeline.
        
        Args:
            visualize: Whether to create visualizations
            n_viz_samples: Number of samples to visualize
            
        Returns:
            Dict with all evaluation results
        """
        print("="*60)
        print("Starting Evaluation")
        print("="*60)
        
        # Run inference
        predictions, targets, external_features = self.predict_all()
        
        # Compute overall metrics
        overall_metrics = self.compute_overall_metrics(predictions, targets)
        
        # Compute subgroup metrics
        subgroup_metrics = self.compute_subgroup_metrics(
            predictions, targets, external_features
        )
        
        # Combine results
        results = {
            "overall": overall_metrics,
            "subgroups": subgroup_metrics
        }
        
        # Save results to JSON
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
        
        # Visualize predictions
        if visualize:
            self.visualize_predictions(predictions, targets, n_viz_samples)
        
        print("\n" + "="*60)
        print("Evaluation Complete!")
        print("="*60)
        
        return results


def load_model(checkpoint_path, use_external=True, t_params=(12, 3, 3), height=32, width=32, device="cpu"):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        use_external: Whether to load extended model (with external features)
        t_params: Temporal parameters
        height: Grid height
        width: Grid width
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model based on type
    if use_external:
        model = create_extended_model(
            t_params=t_params,
            height=height,
            width=width,
            n_external_features=21
        )
    else:
        model = create_baseline_model(
            t_params=t_params,
            height=height,
            width=width
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.6f})")
    
    return model


def main(args):
    """Main evaluation function."""
    print("="*60)
    print("DeepLGR Model Evaluation")
    print("="*60)
    print(f"Configuration:")
    print(f"  Model type: {'Extended (with external features)' if args.use_external else 'Baseline (no external features)'}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Year: {args.year}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {args.device}")
    print()
    
    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = "cpu"
    
    # Load normalization stats
    stats_path = os.path.join(args.data_dir, f"BJ{args.year}_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    print(f"Loaded normalization stats from {stats_path}")
    
    # Load test dataset
    test_path = os.path.join(args.data_dir, f"BJ{args.year}_test.npz")
    test_dataset = TaxiBJDataset(test_path)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Test dataset: {len(test_dataset)} samples, {len(test_loader)} batches\n")
    
    # Load model
    t_params = (args.len_closeness, args.len_period, args.len_trend)
    model = load_model(
        args.checkpoint,
        use_external=args.use_external,
        t_params=t_params,
        device=device
    )
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        stats=stats,
        use_external=args.use_external,
        device=device,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        visualize=args.visualize,
        n_viz_samples=args.n_viz_samples
    )
    
    print("\nEvaluation completed successfully!")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate DeepLGR model (baseline or extended)")
    parser.add_argument("--use_external", action="store_true",
                        help="Use extended model with external features (default: baseline)")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Directory with preprocessed data")
    parser.add_argument("--year", type=str, default="16",
                        help="Year to evaluate (13, 14, 15, or 16)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to evaluate on (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Create visualizations")
    parser.add_argument("--n_viz_samples", type=int, default=5,
                        help="Number of samples to visualize")
    parser.add_argument("--len_closeness", type=int, default=12,
                        help="Number of closeness steps")
    parser.add_argument("--len_period", type=int, default=3,
                        help="Number of period steps")
    parser.add_argument("--len_trend", type=int, default=3,
                        help="Number of trend steps")
    
    args = parser.parse_args()
    main(args)
