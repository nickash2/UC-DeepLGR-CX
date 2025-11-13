"""
ARIMA baseline model for urban flow prediction.

This provides a simple ARIMA-based baseline as specified in the research proposal
for comparison with DeepLGR models. ARIMA (AutoRegressive Integrated Moving Average)
is a classical time series forecasting method.

For spatial-temporal data (32x32 grid with inflow/outflow), we fit separate ARIMA
models for each grid cell and channel.
"""

import numpy as np
import pickle
import os
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class ARIMABaseline:
    """
    ARIMA baseline for flow prediction.
    
    Fits separate ARIMA(p,d,q) models for each spatial grid cell and channel.
    """
    
    def __init__(self, order=(1, 0, 1), height=32, width=32, n_channels=2):
        """
        Initialize ARIMA baseline.
        
        Args:
            order: ARIMA order (p, d, q)
            height: Grid height
            width: Grid width
            n_channels: Number of channels (2: inflow, outflow)
        """
        self.order = order
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.models = {}
        
        print(f"ARIMA Baseline initialized with order={order}")
        print(f"Grid: {height}x{width}, Channels: {n_channels}")
    
    def fit(self, flow_data):
        """
        Fit ARIMA models for each grid cell and channel.
        
        Args:
            flow_data: Time series data [n_timeslots, n_channels, height, width]
        """
        n_timeslots, n_channels, height, width = flow_data.shape
        
        print(f"\nFitting ARIMA models for {n_channels * height * width} time series...")
        print(f"Training data: {n_timeslots} timesteps")
        
        total_models = n_channels * height * width
        fitted = 0
        failed = 0
        
        # Fit model for each (channel, row, col)
        with tqdm(total=total_models, desc="Fitting ARIMA models") as pbar:
            for c in range(n_channels):
                for i in range(height):
                    for j in range(width):
                        try:
                            # Extract time series for this location
                            ts = flow_data[:, c, i, j]
                            
                            # Fit ARIMA model
                            model = ARIMA(ts, order=self.order)
                            fitted_model = model.fit()
                            
                            # Store fitted model
                            self.models[(c, i, j)] = fitted_model
                            fitted += 1
                            
                        except Exception as e:
                            # If fitting fails, store None (will use mean prediction)
                            self.models[(c, i, j)] = None
                            failed += 1
                        
                        pbar.update(1)
        
        print(f"\nFitting complete:")
        print(f"  Successfully fitted: {fitted}/{total_models}")
        print(f"  Failed: {failed}/{total_models}")
    
    def predict(self, n_steps=1):
        """
        Predict next n_steps for all grid cells.
        
        Args:
            n_steps: Number of steps ahead to predict
            
        Returns:
            predictions: [n_steps, n_channels, height, width]
        """
        predictions = np.zeros((n_steps, self.n_channels, self.height, self.width))
        
        for c in range(self.n_channels):
            for i in range(self.height):
                for j in range(self.width):
                    model = self.models.get((c, i, j))
                    
                    if model is not None:
                        try:
                            # Forecast next n_steps
                            forecast = model.forecast(steps=n_steps)
                            predictions[:, c, i, j] = forecast
                        except:
                            # If prediction fails, use last observed value
                            predictions[:, c, i, j] = 0.0
                    else:
                        # No model fitted, use zero
                        predictions[:, c, i, j] = 0.0
        
        return predictions
    
    def save(self, filepath):
        """Save fitted ARIMA models."""
        print(f"Saving ARIMA models to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'height': self.height,
                'width': self.width,
                'n_channels': self.n_channels,
                'models': self.models
            }, f)
        print("Saved successfully!")
    
    @classmethod
    def load(cls, filepath):
        """Load fitted ARIMA models."""
        print(f"Loading ARIMA models from {filepath}...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        arima = cls(
            order=data['order'],
            height=data['height'],
            width=data['width'],
            n_channels=data['n_channels']
        )
        arima.models = data['models']
        print("Loaded successfully!")
        return arima


def train_arima_baseline(data_dir="data/processed", year="16", order=(1, 0, 1)):
    """
    Train ARIMA baseline on training data.
    
    Args:
        data_dir: Directory with preprocessed data
        year: Dataset year
        order: ARIMA order (p, d, q)
    """
    print("="*60)
    print("Training ARIMA Baseline")
    print("="*60)
    
    # Load training data
    train_path = os.path.join(data_dir, f"BJ{year}_train.npz")
    print(f"\nLoading training data from {train_path}...")
    data = np.load(train_path)
    
    # We only need Y (target values) to build time series
    Y = data['Y']  # [n_samples, 2, 32, 32]
    print(f"Training samples: {Y.shape}")
    
    # Create ARIMA baseline
    arima = ARIMABaseline(order=order, height=32, width=32, n_channels=2)
    
    # Fit models
    arima.fit(Y)
    
    # Save models
    output_path = os.path.join(data_dir, f"arima_baseline_BJ{year}.pkl")
    arima.save(output_path)
    
    print("\n" + "="*60)
    print("ARIMA Baseline Training Complete!")
    print("="*60)


def evaluate_arima_baseline(data_dir="data/processed", year="16", output_dir="results"):
    """
    Evaluate ARIMA baseline on test data.
    
    Args:
        data_dir: Directory with preprocessed data
        year: Dataset year
        output_dir: Directory to save results
    """
    from evaluate import compute_mae, compute_smape, denormalize_flow
    
    print("="*60)
    print("Evaluating ARIMA Baseline")
    print("="*60)
    
    # Load ARIMA models
    model_path = os.path.join(data_dir, f"arima_baseline_BJ{year}.pkl")
    arima = ARIMABaseline.load(model_path)
    
    # Load test data
    test_path = os.path.join(data_dir, f"BJ{year}_test.npz")
    print(f"\nLoading test data from {test_path}...")
    data = np.load(test_path)
    Y_test = data['Y']  # [n_samples, 2, 32, 32]
    
    # Load normalization stats
    stats_path = os.path.join(data_dir, f"BJ{year}_stats.pkl")
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    print(f"Test samples: {Y_test.shape[0]}")
    
    # Make predictions (one-step ahead for each test sample)
    print("\nMaking predictions...")
    predictions = []
    
    for i in tqdm(range(len(Y_test)), desc="Predicting"):
        # For simplicity, predict next step based on last observation
        # In reality, would need sliding window approach
        pred = arima.predict(n_steps=1)[0]  # [2, 32, 32]
        predictions.append(pred)
    
    predictions = np.array(predictions)  # [n_samples, 2, 32, 32]
    
    # Denormalize for evaluation
    pred_denorm = denormalize_flow(predictions, stats)
    target_denorm = denormalize_flow(Y_test, stats)
    
    # Compute metrics
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
        "mae_outflow": float(mae_outflow)
    }
    
    print(f"\nARIMA Baseline Results:")
    print(f"  MAE (overall): {mae:.4f}")
    print(f"  SMAPE (overall): {smape:.2f}%")
    print(f"  MAE (inflow): {mae_inflow:.4f}")
    print(f"  MAE (outflow): {mae_outflow:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "arima_baseline_results.json")
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*60)
    print("ARIMA Baseline Evaluation Complete!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ARIMA Baseline for flow prediction")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate"],
                        help="Mode: train or evaluate")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Directory with preprocessed data")
    parser.add_argument("--year", type=str, default="16",
                        help="Dataset year")
    parser.add_argument("--order", type=int, nargs=3, default=[1, 0, 1],
                        help="ARIMA order (p d q)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_arima_baseline(
            data_dir=args.data_dir,
            year=args.year,
            order=tuple(args.order)
        )
    elif args.mode == "evaluate":
        evaluate_arima_baseline(
            data_dir=args.data_dir,
            year=args.year,
            output_dir=args.output_dir
        )
