# DeepLGR Extended - Urban Flow Prediction

Extension of DeepLGR for urban crowd flow prediction with weather and calendar features. Built for the TaxiBJ dataset.

> **Original Implementation**: [yoshall/DeepLGR](https://github.com/yoshall/DeepLGR)

## Project Structure

```
.
├── src/
│   ├── deeplgr.py            # Baseline DeepLGR model (SE blocks + GlobalNet)
│   ├── deeplgr_extended.py   # Extended model with external feature fusion
│   ├── preprocess.py         # General preprocessing pipeline
│   ├── preprocess_periods.py # Period-based preprocessing (P1-P4 from paper)
│   ├── train.py              # Training pipeline with early stopping
│   ├── evaluate.py           # Evaluation with MAE/SMAPE metrics
│   └── arima_baseline.py     # ARIMA baseline for comparison
├── data/
│   ├── BJ{13-16}_M32x32_T30_InOut.h5  # Flow data by year
│   ├── BJ_Meteorology.h5              # Weather data
│   ├── BJ_Holiday.txt                 # Holiday calendar
│   └── processed/                     # Preprocessed .npz files (BJP1-P4)
├── checkpoints_per_period/   # Model checkpoints per period and model type
├── results_per_period/       # Evaluation results per period
├── train_all_periods.sh      # Automated training script for all periods
└── reprocess_and_train.sh    # Full pipeline: preprocess + train + evaluate
```

## Quick Start

### Automated Training (Recommended)

**Train all periods (P1-P4) with baseline and extended models:**
```bash
bash train_all_periods.sh
```

This script automatically:
- Trains baseline and extended models for each period
- Evaluates on test sets 
- Saves results to `results_per_period/all_periods_results.csv`
- Generates checkpoints in `checkpoints_per_period/{period}_{model}/`

### Manual Workflow

### 1. Preprocess Data

**Create P1-P4 datasets (matching original paper):**
```bash
python src/preprocess_periods.py --output_dir data/processed
```

This creates `BJP1_train.npz`, `BJP1_val.npz`, `BJP1_test.npz` (and P2, P3, P4) with:
- Date ranges matching DeepLGR paper Table 1
- 80/10/10 train/val/test split
- 21 external features per sample

### 2. Train Models

**Train extended model (with external features):**
```bash
python src/train.py \
  --use_external \
  --data_path data/processed/BJP1_train.npz \
  --val_path data/processed/BJP1_val.npz \
  --checkpoint_dir checkpoints_per_period/P1_extended \
  --lr 0.0001 \
  --batch_size 16 \
  --epochs 200
```

**Train baseline model (no external features):**
```bash
python src/train.py \
  --data_path data/processed/BJP1_train.npz \
  --val_path data/processed/BJP1_val.npz \
  --checkpoint_dir checkpoints_per_period/P1_baseline \
  --lr 0.0001 \
  --batch_size 16 \
  --epochs 200
```

### 3. Evaluate


```bash
python src/evaluate.py \
  --use_external \
  --data_dir data/processed \
  --year P1 \
  --checkpoint checkpoints_per_period/P1_extended/best.pth \
  --output_dir results_per_period/P1_extended
```

This generates:
- `evaluation_results.json` with overall metrics
- Sample visualization PNGs

## Model Details

### DeepLGR Baseline

Standard architecture from the original paper:
- 9 SE (Squeeze-and-Excitation) residual blocks
- GlobalNet with multi-scale pooling (1x1, 2x2, 4x4, 8x8)
- Tensor decomposition predictor
- Input: Closeness (12), Period (3), Trend (3) temporal windows

### DeepLGR Extended (ST-ResNet-style Fusion)

Modified architecture with late-fusion external features:
- **Flow path**: Same backbone as baseline (SENet + GlobalNet + Predictor)
- **External path**: Separate 2-layer MLP (21 $\rightarrow$ 10 $\rightarrow$ 2048 features)
- **Fusion**: Element-wise addition at output (residual connection)
  ```
  output = flow_prediction + external_embedding
  ```


**External Features (21 total):**
- Temperature (1, z-score normalized)
- Wind speed (1, z-score normalized)
- Weather one-hot (17 categories)
- Weekend flag (1, binary)
- Holiday flag (1, binary)

## Data Format

**Flow Data:**
- Shape: `[batch, 2, 32, 32]`
- Channels: inflow, outflow
- Grid: 32x32 Beijing representation
- Interval: 30 minutes, filtered to 6am-11pm (slots 12-46)
- Step size: 1 timeslot (overlapping samples)

**Temporal Input:**
- Closeness: 12 consecutive recent timeslots (6 hours)
- Period: 3 timeslots at same time from previous 3 days
- Trend: 3 timeslots at same time from previous 3 weeks
- Total lookback: Up to 21 days
- Prediction: 1 timeslot ahead (30 minutes)

**Preprocessed Files:**
```
BJP{1-4}_train.npz:
  - X_closeness: [n, 12, 2, 32, 32]
  - X_period: [n, 3, 2, 32, 32]
  - X_trend: [n, 3, 2, 32, 32]
  - X_external: [n, 21]
  - Y: [n, 2, 32, 32]
  - timestamps: [n]
```

## Training Details

**Default hyperparameters:**
- Optimizer: Adam (lr=0.005)
- Loss: MSE
- Batch size: 32
- Early stopping: patience=10
- LR scheduling: ReduceLROnPlateau (factor=0.5, patience=5)

**Checkpointing:**
- `best.pth`: Best validation loss
- `latest.pth`: Most recent epoch
- `history.json`: Training/validation loss curves


## References

- **Original DeepLGR**: [yoshall/DeepLGR](https://github.com/yoshall/DeepLGR)
- **Paper**: "Revisiting Convolutional Neural Networks for Citywide Crowd Flow Analytics" ([Arxiv URL](https://arxiv.org/abs/2003.00895))
- **Dataset**: "TaxiBJ: InFlow/OutFlow, Meteorology and Holidays at Beijing" ([Repo URL](https://gitee.com/arislee/taxi-bj))

