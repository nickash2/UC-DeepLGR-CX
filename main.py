import os
import sys
import argparse
import subprocess


def run_preprocessing(args):
    """Run preprocessing step."""
    print("\n" + "="*70)
    print("STEP 1: PREPROCESSING")
    print("="*70 + "\n")
    
    cmd = [
        sys.executable, "src/preprocess.py",
        "--years", args.years,
        "--data_dir", args.data_dir,
        "--output_dir", args.processed_dir,
        "--len_closeness", str(args.len_closeness),
        "--len_period", str(args.len_period),
        "--len_trend", str(args.len_trend)
    ]
    
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def run_training(args):
    """Run training step."""
    print("\n" + "="*70)
    print("STEP 2: TRAINING")
    print("="*70 + "\n")
    
    cmd = [
        sys.executable, "src/train.py",
        "--data_dir", args.processed_dir,
        "--year", args.years,
        "--checkpoint_dir", args.checkpoint_dir,
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--device", args.device,
        "--num_workers", str(args.num_workers),
        "--len_closeness", str(args.len_closeness),
        "--len_period", str(args.len_period),
        "--len_trend", str(args.len_trend)
    ]
    
    # Add external features flag if requested
    if args.use_external:
        cmd.append("--use_external")
    
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def run_evaluation(args):
    """Run evaluation step."""
    print("\n" + "="*70)
    print("STEP 3: EVALUATION")
    print("="*70 + "\n")
    
    cmd = [
        sys.executable, "src/evaluate.py",
        "--data_dir", args.processed_dir,
        "--year", args.years,
        "--checkpoint", args.checkpoint,
        "--output_dir", args.results_dir,
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--num_workers", str(args.num_workers),
        "--n_viz_samples", str(args.n_viz_samples),
        "--len_closeness", str(args.len_closeness),
        "--len_period", str(args.len_period),
        "--len_trend", str(args.len_trend)
    ]
    
    # Add flags
    if args.use_external:
        cmd.append("--use_external")
    if args.visualize:
        cmd.append("--visualize")
    
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def run_arima_baseline(args):
    """Run ARIMA baseline training and evaluation."""
    print("\n" + "="*70)
    print("ARIMA BASELINE")
    print("="*70 + "\n")
    
    # Train ARIMA
    print("Training ARIMA baseline...")
    train_cmd = [
        sys.executable, "src/arima_baseline.py",
        "--mode", "train",
        "--data_dir", args.processed_dir,
        "--year", args.years
    ]
    result = subprocess.run(train_cmd, check=True)
    if result.returncode != 0:
        return False
    
    # Evaluate ARIMA
    print("\nEvaluating ARIMA baseline...")
    eval_cmd = [
        sys.executable, "src/arima_baseline.py",
        "--mode", "evaluate",
        "--data_dir", args.processed_dir,
        "--year", args.years,
        "--output_dir", args.results_dir
    ]
    result = subprocess.run(eval_cmd, check=True)
    return result.returncode == 0


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="DeepLGR Urban Flow Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline for BJ16 dataset
  python main.py --mode all --years 16

  # Run complete pipeline for all years (2013-2016)
  python main.py --mode all --years 13-16 --use_external

  # Run only preprocessing for multiple years
  python main.py --mode preprocess --years 13,14,15,16

  # Run training with custom parameters on multi-year data
  python main.py --mode train --years 13-16 --epochs 50 --batch_size 64

  # Run evaluation with visualization
  python main.py --mode evaluate --years 13-16 --visualize
        """
    )
    
    # Pipeline mode
    parser.add_argument("--mode", type=str, required=True,
                        choices=["all", "preprocess", "train", "evaluate", "arima"],
                        help="Pipeline mode: all (run complete pipeline), "
                             "preprocess, train, evaluate, or arima (ARIMA baseline)")
    
    # Model configuration
    parser.add_argument("--use_external", action="store_true",
                        help="Use extended model with external contextual features "
                             "(default: baseline model without external features)")
    
    # Data parameters
    parser.add_argument("--years", type=str, default="16",
                        help="TaxiBJ year(s) to process: single (16), comma-separated (13,14,15,16), or range (13-16)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory with raw data files")
    parser.add_argument("--processed_dir", type=str, default="data/processed",
                        help="Directory for preprocessed data")
    
    # Model parameters
    parser.add_argument("--len_closeness", type=int, default=12,
                        help="Number of closeness steps (recent consecutive)")
    parser.add_argument("--len_period", type=int, default=3,
                        help="Number of period steps (daily pattern)")
    parser.add_argument("--len_trend", type=int, default=3,
                        help="Number of trend steps (weekly pattern)")
    
    # Training parameters
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training/evaluation")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    
    # Evaluation parameters
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth",
                        help="Path to model checkpoint for evaluation")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save evaluation results")
    parser.add_argument("--visualize", action="store_true",
                        help="Create visualization plots during evaluation")
    parser.add_argument("--n_viz_samples", type=int, default=5,
                        help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    print("="*70)
    print("DeepLGR URBAN FLOW PREDICTION PIPELINE")
    print("="*70)
    print(f"\nMode: {args.mode}")
    print(f"Model: {'Extended (with external features)' if args.use_external else 'Baseline (no external features)'}")
    print(f"Dataset: BJ{args.years}")
    print(f"Temporal parameters: closeness={args.len_closeness}, "
          f"period={args.len_period}, trend={args.len_trend}")
    print()
    
    try:
        if args.mode == "all":
            # Run complete pipeline including ARIMA baseline
            print("Running complete pipeline: Preprocess → Train → Evaluate → ARIMA Baseline\n")
            
            success = run_preprocessing(args)
            if not success:
                print("Error: Preprocessing failed")
                return 1
            
            success = run_training(args)
            if not success:
                print("Error: Training failed")
                return 1
            
            success = run_evaluation(args)
            if not success:
                print("Error: Evaluation failed")
                return 1
            
            # Run ARIMA baseline for comparison
            success = run_arima_baseline(args)
            if not success:
                print("Error: ARIMA baseline failed")
                return 1
            
            print("\n" + "="*70)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"\nResults saved to: {args.results_dir}")
            print(f"  - DeepLGR results: evaluation_results.json")
            print(f"  - ARIMA baseline: arima_baseline_results.json")
            print(f"Model checkpoints: {args.checkpoint_dir}")
            
        elif args.mode == "preprocess":
            success = run_preprocessing(args)
            if success:
                print(f"\nPreprocessing complete! Data saved to: {args.processed_dir}")
            return 0 if success else 1
            
        elif args.mode == "train":
            success = run_training(args)
            if success:
                print(f"\nTraining complete! Checkpoints saved to: {args.checkpoint_dir}")
            return 0 if success else 1
            
        elif args.mode == "evaluate":
            success = run_evaluation(args)
            if success:
                print(f"\nEvaluation complete! Results saved to: {args.results_dir}")
            return 0 if success else 1
        
        elif args.mode == "arima":
            success = run_arima_baseline(args)
            if success:
                print(f"\nARIMA baseline complete! Results saved to: {args.results_dir}")
            return 0 if success else 1
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: Pipeline step failed with exit code {e.returncode}")
        return 1
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
