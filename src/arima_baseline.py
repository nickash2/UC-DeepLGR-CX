import numpy as np
import pickle
import os
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class ARIMABaseline:
    def __init__(self, order=(1, 0, 1), height=32, width=32, n_channels=2):
        self.order = order
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.models = {}

        print(f"ARIMA Baseline initialized with order={order}")
        print(f"Grid: {height}x{width}, Channels: {n_channels}")

    def fit(self, flow_data):
        n_timeslots, n_channels, height, width = flow_data.shape

        print(
            f"\nFitting ARIMA models for {n_channels * height * width} time series..."
        )
        print(f"Training data: {n_timeslots} timesteps")

        total_models = n_channels * height * width
        fitted = 0
        failed = 0

        with tqdm(total=total_models, desc="Fitting ARIMA models") as pbar:
            for c in range(n_channels):
                for i in range(height):
                    for j in range(width):
                        try:
                            ts = flow_data[:, c, i, j]
                            model = ARIMA(ts, order=self.order)
                            fitted_model = model.fit()
                            self.models[(c, i, j)] = fitted_model
                            fitted += 1
                        except Exception as e:  # noqa: F841
                            self.models[(c, i, j)] = None
                            failed += 1
                        pbar.update(1)

        print("\nFitting complete:")
        print(f"  Successfully fitted: {fitted}/{total_models}")
        print(f"  Failed: {failed}/{total_models}")

    def predict(self, n_steps=1):
        predictions = np.zeros((n_steps, self.n_channels, self.height, self.width))

        for c in range(self.n_channels):
            for i in range(self.height):
                for j in range(self.width):
                    model = self.models.get((c, i, j))

                    if model is not None:
                        try:
                            forecast = model.forecast(steps=n_steps)
                            predictions[:, c, i, j] = forecast
                        except:  # noqa: E722
                            predictions[:, c, i, j] = 0.0
                    else:
                        predictions[:, c, i, j] = 0.0

        return predictions

    def save(self, filepath):
        print(f"Saving ARIMA models to {filepath}...")
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "order": self.order,
                    "height": self.height,
                    "width": self.width,
                    "n_channels": self.n_channels,
                    "models": self.models,
                },
                f,
            )
        print("Saved successfully!")

    @classmethod
    def load(cls, filepath):
        print(f"Loading ARIMA models from {filepath}...")
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        arima = cls(
            order=data["order"],
            height=data["height"],
            width=data["width"],
            n_channels=data["n_channels"],
        )
        arima.models = data["models"]
        print("Loaded successfully!")
        return arima


def train_arima_baseline(data_dir="data/processed", year="16", order=(1, 0, 1)):
    print("=" * 60)
    print("Training ARIMA Baseline")
    print("=" * 60)

    train_path = os.path.join(data_dir, f"BJ{year}_train.npz")
    print(f"\nLoading training data from {train_path}...")
    data = np.load(train_path)

    Y = data["Y"]
    print(f"Training samples: {Y.shape}")

    arima = ARIMABaseline(order=order, height=32, width=32, n_channels=2)
    arima.fit(Y)

    output_path = os.path.join(data_dir, f"arima_baseline_BJ{year}.pkl")
    arima.save(output_path)

    print("\n" + "=" * 60)
    print("ARIMA Baseline Training Complete!")
    print("=" * 60)


def evaluate_arima_baseline(data_dir="data/processed", year="16", output_dir="results"):
    from evaluate import compute_mae, compute_smape, denormalize_flow

    print("=" * 60)
    print("Evaluating ARIMA Baseline")
    print("=" * 60)

    model_path = os.path.join(data_dir, f"arima_baseline_BJ{year}.pkl")
    arima = ARIMABaseline.load(model_path)

    test_path = os.path.join(data_dir, f"BJ{year}_test.npz")
    print(f"\nLoading test data from {test_path}...")
    data = np.load(test_path)
    Y_test = data["Y"]

    stats_path = os.path.join(data_dir, f"BJ{year}_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    print(f"Test samples: {Y_test.shape[0]}")

    print("\nMaking predictions...")
    predictions = []

    for i in tqdm(range(len(Y_test)), desc="Predicting"):
        pred = arima.predict(n_steps=1)[0]
        predictions.append(pred)

    predictions = np.array(predictions)

    pred_denorm = denormalize_flow(predictions, stats)
    target_denorm = denormalize_flow(Y_test, stats)

    print("\nComputing metrics...")
    mae = compute_mae(pred_denorm, target_denorm)
    smape = compute_smape(pred_denorm, target_denorm)

    mae_inflow = compute_mae(pred_denorm[:, 0], target_denorm[:, 0])
    mae_outflow = compute_mae(pred_denorm[:, 1], target_denorm[:, 1])

    results = {
        "model": "ARIMA Baseline",
        "order": arima.order,
        "mae_overall": float(mae),
        "smape_overall": float(smape),
        "mae_inflow": float(mae_inflow),
        "mae_outflow": float(mae_outflow),
    }

    print("\nARIMA Baseline Results:")
    print(f"  MAE (overall): {mae:.4f}")
    print(f"  SMAPE (overall): {smape:.2f}%")
    print(f"  MAE (inflow): {mae_inflow:.4f}")
    print(f"  MAE (outflow): {mae_outflow:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "arima_baseline_results.json")
    import json

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 60)
    print("ARIMA Baseline Evaluation Complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARIMA Baseline for flow prediction")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "evaluate"],
        help="Mode: train or evaluate",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory with preprocessed data",
    )
    parser.add_argument("--year", type=str, default="16", help="Dataset year")
    parser.add_argument(
        "--order", type=int, nargs=3, default=[1, 0, 1], help="ARIMA order (p d q)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory for results"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_arima_baseline(
            data_dir=args.data_dir, year=args.year, order=tuple(args.order)
        )
    elif args.mode == "evaluate":
        evaluate_arima_baseline(
            data_dir=args.data_dir, year=args.year, output_dir=args.output_dir
        )
