import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import json
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deeplgr_extended import create_baseline_model, create_extended_model
from train import TaxiBJDataset


def denormalize_flow(flow_data, stats):
    if "flow_mean" in stats and "flow_std" in stats:
        flow_mean = stats["flow_mean"]
        flow_std = stats["flow_std"]
        return flow_data * flow_std + flow_mean
    else:
        flow_min = stats["flow_min"]
        flow_max = stats["flow_max"]
        return flow_data * (flow_max - flow_min) + flow_min


def compute_mae(predictions, targets):
    z = predictions.size
    mae = (1.0 / z) * np.sum(np.abs(targets - predictions))
    return mae


def compute_smape(predictions, targets):
    z = predictions.size
    numerator = np.abs(targets - predictions)
    denominator = np.abs(targets) + np.abs(predictions)

    mask = denominator > 0
    smape_values = np.zeros_like(numerator)
    smape_values[mask] = numerator[mask] / denominator[mask]

    smape = (1.0 / z) * np.sum(smape_values)
    return smape * 100


class Evaluator:
    def __init__(
        self,
        model,
        test_loader,
        stats,
        use_external=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir="results",
    ):
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
        print("Running inference on test set...")

        all_predictions = []
        all_targets = []
        all_external = []

        with torch.no_grad():
            for batch_idx, (X_c, X_p, X_t, X_ext, Y) in enumerate(self.test_loader):
                X_c = X_c.to(self.device)
                X_p = X_p.to(self.device)
                X_t = X_t.to(self.device)
                X_ext = X_ext.to(self.device)
                Y = Y.to(self.device)

                b = X_c.shape[0]
                X_c_flat = X_c.reshape(b, -1, X_c.shape[-2], X_c.shape[-1])
                X_p_flat = X_p.reshape(b, -1, X_p.shape[-2], X_p.shape[-1])
                X_t_flat = X_t.reshape(b, -1, X_t.shape[-2], X_t.shape[-1])

                if self.use_external:
                    prediction = self.model((X_c_flat, X_p_flat, X_t_flat), X_ext)
                else:
                    prediction = self.model((X_c_flat, X_p_flat, X_t_flat))

                all_predictions.append(prediction.cpu().numpy())
                all_targets.append(Y.cpu().numpy())
                all_external.append(X_ext.cpu().numpy())

                if (batch_idx + 1) % 50 == 0:
                    print(
                        f"  Processed {batch_idx + 1}/{len(self.test_loader)} batches"
                    )

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        external_features = np.concatenate(all_external, axis=0)

        print(f"Inference complete: {len(predictions)} samples")
        return predictions, targets, external_features

    def compute_overall_metrics(self, predictions, targets):
        print("\nComputing overall metrics...")

        pred_denorm = denormalize_flow(predictions, self.stats)
        target_denorm = denormalize_flow(targets, self.stats)

        mae = compute_mae(pred_denorm, target_denorm)
        smape = compute_smape(pred_denorm, target_denorm)

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
            "smape_outflow": float(smape_outflow),
        }

        print(f"  MAE (overall): {mae:.4f}")
        print(f"  SMAPE (overall): {smape:.2f}%")
        print(f"  MAE (inflow): {mae_inflow:.4f}")
        print(f"  MAE (outflow): {mae_outflow:.4f}")

        return metrics

    def compute_subgroup_metrics(self, predictions, targets, external_features):
        print("\nComputing subgroup metrics...")

        pred_denorm = denormalize_flow(predictions, self.stats)
        target_denorm = denormalize_flow(targets, self.stats)

        subgroup_metrics = {}

        is_weekend = external_features[:, 19] > 0.5
        is_weekday = ~is_weekend

        if np.sum(is_weekend) > 0:
            mae_weekend = compute_mae(
                pred_denorm[is_weekend], target_denorm[is_weekend]
            )
            smape_weekend = compute_smape(
                pred_denorm[is_weekend], target_denorm[is_weekend]
            )
            subgroup_metrics["weekend"] = {
                "count": int(np.sum(is_weekend)),
                "mae": float(mae_weekend),
                "smape": float(smape_weekend),
            }
            print(
                f"  Weekend ({np.sum(is_weekend)} samples): MAE={mae_weekend:.4f}, SMAPE={smape_weekend:.2f}%"
            )

        if np.sum(is_weekday) > 0:
            mae_weekday = compute_mae(
                pred_denorm[is_weekday], target_denorm[is_weekday]
            )
            smape_weekday = compute_smape(
                pred_denorm[is_weekday], target_denorm[is_weekday]
            )
            subgroup_metrics["weekday"] = {
                "count": int(np.sum(is_weekday)),
                "mae": float(mae_weekday),
                "smape": float(smape_weekday),
            }
            print(
                f"  Weekday ({np.sum(is_weekday)} samples): MAE={mae_weekday:.4f}, SMAPE={smape_weekday:.2f}%"
            )

        is_holiday = external_features[:, 20] > 0.5
        is_regular = ~is_holiday

        if np.sum(is_holiday) > 0:
            mae_holiday = compute_mae(
                pred_denorm[is_holiday], target_denorm[is_holiday]
            )
            smape_holiday = compute_smape(
                pred_denorm[is_holiday], target_denorm[is_holiday]
            )
            subgroup_metrics["holiday"] = {
                "count": int(np.sum(is_holiday)),
                "mae": float(mae_holiday),
                "smape": float(smape_holiday),
            }
            print(
                f"  Holiday ({np.sum(is_holiday)} samples): MAE={mae_holiday:.4f}, SMAPE={smape_holiday:.2f}%"
            )

        if np.sum(is_regular) > 0:
            mae_regular = compute_mae(
                pred_denorm[is_regular], target_denorm[is_regular]
            )
            smape_regular = compute_smape(
                pred_denorm[is_regular], target_denorm[is_regular]
            )
            subgroup_metrics["regular"] = {
                "count": int(np.sum(is_regular)),
                "mae": float(mae_regular),
                "smape": float(smape_regular),
            }
            print(
                f"  Regular ({np.sum(is_regular)} samples): MAE={mae_regular:.4f}, SMAPE={smape_regular:.2f}%"
            )

        weather_one_hot = external_features[:, 2:19]
        rainy_indices = [3, 4, 5, 6, 7, 8, 9]
        is_rainy = weather_one_hot[:, rainy_indices].sum(axis=1) > 0.5
        is_sunny = ~is_rainy

        if np.sum(is_rainy) > 0:
            mae_rainy = compute_mae(pred_denorm[is_rainy], target_denorm[is_rainy])
            smape_rainy = compute_smape(pred_denorm[is_rainy], target_denorm[is_rainy])
            subgroup_metrics["rainy"] = {
                "count": int(np.sum(is_rainy)),
                "mae": float(mae_rainy),
                "smape": float(smape_rainy),
            }
            print(
                f"  Rainy ({np.sum(is_rainy)} samples): MAE={mae_rainy:.4f}, SMAPE={smape_rainy:.2f}%"
            )

        if np.sum(is_sunny) > 0:
            mae_sunny = compute_mae(pred_denorm[is_sunny], target_denorm[is_sunny])
            smape_sunny = compute_smape(pred_denorm[is_sunny], target_denorm[is_sunny])
            subgroup_metrics["sunny"] = {
                "count": int(np.sum(is_sunny)),
                "mae": float(mae_sunny),
                "smape": float(smape_sunny),
            }
            print(
                f"  Non-rainy ({np.sum(is_sunny)} samples): MAE={mae_sunny:.4f}, SMAPE={smape_sunny:.2f}%"
            )

        return subgroup_metrics

    def visualize_predictions(self, predictions, targets, n_samples=5):
        print(f"\nCreating visualizations for {n_samples} samples...")

        pred_denorm = denormalize_flow(predictions, self.stats)
        target_denorm = denormalize_flow(targets, self.stats)

        pred_denorm = denormalize_flow(predictions, self.stats)
        target_denorm = denormalize_flow(targets, self.stats)

        indices = np.random.choice(len(predictions), n_samples, replace=False)

        for idx in indices:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            im0 = axes[0, 0].imshow(
                target_denorm[idx, 0], cmap="hot", interpolation="nearest"
            )
            axes[0, 0].set_title("Inflow - Ground Truth")
            axes[0, 0].axis("off")
            plt.colorbar(im0, ax=axes[0, 0])

            im1 = axes[0, 1].imshow(
                pred_denorm[idx, 0], cmap="hot", interpolation="nearest"
            )
            axes[0, 1].set_title("Inflow - Prediction")
            axes[0, 1].axis("off")
            plt.colorbar(im1, ax=axes[0, 1])

            im2 = axes[1, 0].imshow(
                target_denorm[idx, 1], cmap="hot", interpolation="nearest"
            )
            axes[1, 0].set_title("Outflow - Ground Truth")
            axes[1, 0].axis("off")
            plt.colorbar(im2, ax=axes[1, 0])

            im3 = axes[1, 1].imshow(
                pred_denorm[idx, 1], cmap="hot", interpolation="nearest"
            )
            axes[1, 1].set_title("Outflow - Prediction")
            axes[1, 1].axis("off")
            plt.colorbar(im3, ax=axes[1, 1])

            mae_sample = compute_mae(pred_denorm[idx], target_denorm[idx])
            plt.suptitle(f"Sample {idx} - MAE: {mae_sample:.4f}")

            plt.tight_layout()

            fig_path = os.path.join(self.output_dir, f"sample_{idx}_visualization.png")
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()

        print(f"Visualizations saved to {self.output_dir}")

    def evaluate(self, visualize=True, n_viz_samples=5):
        print("=" * 60)
        print("Starting Evaluation")
        print("=" * 60)

        predictions, targets, external_features = self.predict_all()

        overall_metrics = self.compute_overall_metrics(predictions, targets)

        subgroup_metrics = self.compute_subgroup_metrics(
            predictions, targets, external_features
        )

        results = {"overall": overall_metrics, "subgroups": subgroup_metrics}

        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

        if visualize:
            self.visualize_predictions(predictions, targets, n_viz_samples)

        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)

        return results


def load_model(
    checkpoint_path,
    use_external=True,
    t_params=(12, 3, 3),
    height=32,
    width=32,
    device="cpu",
):
    print(f"Loading model from {checkpoint_path}...")

    if use_external:
        model = create_extended_model(
            t_params=t_params, height=height, width=width, n_external_features=21
        )
    else:
        model = create_baseline_model(t_params=t_params, height=height, width=width)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.6f})"
    )

    return model


def main(args):
    print("=" * 60)
    print("DeepLGR Model Evaluation")
    print("=" * 60)

    if "," in args.year or "-" in args.year:
        if "-" in args.year and "," not in args.year:
            start, end = args.year.split("-")
            year_suffix = f"{start.zfill(2)}-{end.zfill(2)}"
        else:
            years = [y.strip().zfill(2) for y in args.year.split(",")]
            year_suffix = f"{years[0]}-{years[-1]}"
    else:
        year_suffix = args.year.zfill(2)

    print("Configuration:")
    print(
        f"  Model type: {'Extended (with external features)' if args.use_external else 'Baseline (no external features)'}"
    )
    print(f"  Data directory: {args.data_dir}")
    print(f"  Year(s): {args.year} -> BJ{year_suffix}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {args.device}")
    print()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = "cpu"

    stats_path = os.path.join(args.data_dir, f"BJ{year_suffix}_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    print(f"Loaded normalization stats from {stats_path}")

    test_path = os.path.join(args.data_dir, f"BJ{year_suffix}_test.npz")
    test_dataset = TaxiBJDataset(test_path)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Test dataset: {len(test_dataset)} samples, {len(test_loader)} batches\n")

    t_params = (args.len_closeness, args.len_period, args.len_trend)
    model = load_model(
        args.checkpoint,
        use_external=args.use_external,
        t_params=t_params,
        device=device,
    )

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        stats=stats,
        use_external=args.use_external,
        device=device,
        output_dir=args.output_dir,
    )

    results = evaluator.evaluate(
        visualize=args.visualize, n_viz_samples=args.n_viz_samples
    )

    print("\nEvaluation completed successfully!")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate DeepLGR model (baseline or extended)"
    )
    parser.add_argument(
        "--use_external",
        action="store_true",
        help="Use extended model with external features (default: baseline)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory with preprocessed data",
    )
    parser.add_argument(
        "--year",
        type=str,
        default="16",
        help="Year(s) to evaluate: single (16), comma-separated (13,14,15,16), or range (13-16)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to evaluate on (cuda or cpu)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--visualize", action="store_true", default=True, help="Create visualizations"
    )
    parser.add_argument(
        "--n_viz_samples", type=int, default=5, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--len_closeness", type=int, default=12, help="Number of closeness steps"
    )
    parser.add_argument(
        "--len_period", type=int, default=3, help="Number of period steps"
    )
    parser.add_argument(
        "--len_trend", type=int, default=3, help="Number of trend steps"
    )

    args = parser.parse_args()
    main(args)
