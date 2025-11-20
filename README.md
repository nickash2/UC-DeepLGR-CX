# DeepLGR Extended - Urban Flow Prediction

Extension of DeepLGR for urban crowd flow prediction with weather and calendar features. Built for the TaxiBJ dataset.

> **Original Implementation**: [yoshall/DeepLGR](https://github.com/yoshall/DeepLGR)

## Project Structure

```
.
├── src/
│   ├── deeplgr.py            # Baseline DeepLGR model (SE blocks + GlobalNet)
│   ├── deeplgr_extended.py   # Extended model with external features
│   ├── preprocess.py         # General preprocessing pipeline
│   ├── preprocess_periods.py # Period-based preprocessing (P1-P4 from paper)
│   ├── train.py              # Training pipeline with early stopping
│   ├── evaluate.py           # Evaluation with MAE/SMAPE metrics
│   └── arima_baseline.py     # ARIMA baseline for comparison
├── data/
│   ├── BJ{13-16}_M32x32_T30_InOut.h5  # Flow data by year
│   ├── BJ_Meteorology.h5              # Weather data
│   ├── BJ_Holiday.txt                 # Holiday calendar
│   └── processed/                     # Preprocessed .npz files
├── checkpoints/              # Model checkpoints
└── results/                  # Evaluation results
```

## Quick Start

### 1. Preprocess Data

**Using period-based splits (recommended for paper comparison):**
```bash
python src/preprocess_periods.py --output_dir data/processed
```

This creates P1-P4 datasets matching the original DeepLGR paper date ranges.

**Or preprocess single year:**
```bash
python src/preprocess.py --year 16 --output_dir data/processed
```

### 2. Train Models

**Train extended model (with external features):**
```bash
python src/train.py \
  --use_external \
  --data_path data/processed/BJP1_train.npz \
  --val_path data/processed/BJP1_val.npz \
  --checkpoint_dir checkpoints_extended \
  --epochs 100
```

**Train baseline model (no external features):**
```bash
python src/train.py \
  --data_path data/processed/BJP1_train.npz \
  --val_path data/processed/BJP1_val.npz \
  --checkpoint_dir checkpoints_baseline \
  --epochs 100
```

### 3. Evaluate

**Evaluate extended model:**
```bash
python src/evaluate.py \
  --use_external \
  --data_dir data/processed \
  --year P1 \
  --checkpoint checkpoints_extended/best.pth \
  --output_dir results
```

**Evaluate baseline:**
```bash
python src/evaluate.py \
  --data_dir data/processed \
  --year P1 \
  --checkpoint checkpoints_baseline/best.pth \
  --output_dir results
```

### 4. Run ARIMA Baseline

**Train ARIMA:**
```bash
python src/arima_baseline.py --mode train --data_dir data/processed --year P1
```

**Evaluate ARIMA:**
```bash
python src/arima_baseline.py --mode evaluate --data_dir data/processed --year P1
```

## Model Details

### DeepLGR Baseline

Standard architecture from the original paper:
- 9 SE (Squeeze-and-Excitation) residual blocks
- GlobalNet with multi-scale pooling (1×1, 2×2, 4×4, 8×8)
- Tensor decomposition predictor
- Input: Closeness (12), Period (3), Trend (3) temporal windows

### DeepLGR Extended

Modified architecture with external features:
- Same backbone as baseline
- Additional input: 21 external features (weather + calendar)
- Features broadcast spatially to 32×32 grid
- Concatenated with flow data before first conv layer

**External Features (21 total):**
- Temperature (1, normalized)
- Wind speed (1, normalized)
- Weather one-hot (17 categories)
- Weekend flag (1, binary)
- Holiday flag (1, binary)

## Data Format

**Flow Data:**
- Shape: `[batch, 2, 32, 32]`
- Channels: inflow, outflow
- Grid: 32×32 Beijing representation
- Interval: 30 minutes, filtered to 6am-11pm

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

## Evaluation Metrics

**Overall:**
- MAE (Mean Absolute Error)
- SMAPE (Symmetric Mean Absolute Percentage Error)

**Subgroup Analysis:**
- Weekday vs Weekend
- Regular vs Holiday
- Rainy vs Non-rainy

All metrics computed separately for inflow and outflow.

## Preprocessing Options

**Period-based (preprocess_periods.py):**
- P1: 2013-07-01 to 2013-10-30
- P2: 2014-03-01 to 2014-06-27
- P3: 2015-03-01 to 2015-06-30
- P4: 2015-11-01 to 2016-04-10

**General (preprocess.py):**
```bash
python src/preprocess.py \
  --year 16 \
  --len_closeness 12 \
  --len_period 3 \
  --len_trend 3 \
  --output_dir data/processed
```

## References

- **Original DeepLGR**: [yoshall/DeepLGR](https://github.com/yoshall/DeepLGR)
- **Paper**: "Revisiting Convolutional Neural Networks for Citywide Crowd Flow Analytics" ([Arxiv URL](https://arxiv.org/abs/2003.00895))
- **Dataset**: "TaxiBJ: InFlow/OutFlow, Meteorology and Holidays at Beijing" ([Repo URL](https://gitee.com/arislee/taxi-bj))

