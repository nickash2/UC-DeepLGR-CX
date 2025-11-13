# DeepLGR Urban Flow Prediction with External Features
# delete the instructions under .github before submission

Research project extending DeepLGR for urban crowd flow prediction by integrating weather and calendar features into spatio-temporal predictions.

## Project Structure

```
.
├── main.py                  # Master pipeline orchestration script
├── src/
│   ├── deeplgr.py          # DeepLGR model architecture (SE blocks + GlobalNet)
│   ├── preprocess.py       # Data preprocessing (TaxiBJ + weather + calendar)
│   ├── train.py            # Model training with checkpointing
│   └── evaluate.py         # Evaluation with MAE/SMAPE metrics
├── data/                    # Raw TaxiBJ dataset
│   ├── BJ13-16_M32x32_T30_InOut.h5
│   ├── BJ_Meteorology.h5
│   └── BJ_Holiday.txt
└── checkpoints/             # Saved model checkpoints (created during training)
```

## Quick Start

### 1. Run Complete Pipeline

Process data, train model, and evaluate results:

```bash
python main.py --mode all --year 16 --epochs 100
```

### 2. Run Individual Steps

**Preprocessing only:**
```bash
python main.py --mode preprocess --year 16
```

**Training only:**
```bash
python main.py --mode train --year 16 --epochs 50 --batch_size 64
```

**Evaluation only:**
```bash
python main.py --mode evaluate --year 16 --visualize
```

## Pipeline Details

### Step 1: Preprocessing (`src/preprocess.py`)

Prepares TaxiBJ dataset with external features:

- **Filters timeslots**: 6am-11pm (slots 12-45 of 48 daily slots)
- **Integrates weather**: Temperature, wind speed, 17 weather categories (one-hot)
- **Integrates calendar**: Weekend/weekday flags, holiday indicators
- **Creates temporal samples**: Closeness (12 recent), Period (3 daily), Trend (3 weekly)
- **Splits data**: 80% train, 10% validation, 10% test
- **Normalizes**: Flow data (min-max), continuous features (z-score)

**Output:** `data/processed/BJ{year}_{train,val,test}.npz` + normalization stats

**Example:**
```bash
python src/preprocess.py --year 16 --len_closeness 12 --len_period 3 --len_trend 3
```

### Step 2: Training (`src/train.py`)

Trains DeepLGR model with:

- **Architecture**: 9 SE residual blocks + GlobalNet + tensor decomposition predictor
- **Optimizer**: Adam (default lr=0.001)
- **Loss**: MSE (Mean Squared Error)
- **Early stopping**: Patience=10 epochs
- **Checkpointing**: Saves best model by validation loss

**Output:** `checkpoints/best.pth`, `checkpoints/latest.pth`, `checkpoints/history.json`

**Example:**
```bash
python src/train.py --data_dir data/processed --year 16 --epochs 100 --batch_size 32
```

**Note:** External features are loaded but not yet integrated into the model architecture. Integration is the research contribution to implement.

### Step 3: Evaluation (`src/evaluate.py`)

Evaluates trained model with:

- **Overall metrics**: MAE, SMAPE for inflow/outflow
- **Subgroup analysis**:
  - Weekday vs Weekend
  - Regular day vs Holiday
  - Rainy vs Non-rainy weather
- **Visualizations**: Sample predictions vs ground truth heatmaps

**Output:** `results/evaluation_results.json`, visualization PNGs

**Example:**
```bash
python src/evaluate.py --checkpoint checkpoints/best.pth --year 16 --visualize
```

## Model Architecture

DeepLGR consists of:

1. **Input**: Three temporal components (closeness, period, trend)
   - Closeness: Recent 12 consecutive timeslots
   - Period: Same time from previous 3 days (daily pattern)
   - Trend: Same time from previous 3 weeks (weekly pattern)

2. **SENet Path**: Conv → 9× SE Residual Blocks → Conv (with skip connection)
   - SE (Squeeze-and-Excitation) blocks for local spatial relations

3. **GlobalNet**: Multi-scale pyramid pooling + SubPixel upsampling
   - Captures global spatial dependencies across the city grid

4. **Predictor**: Tensor decomposition (default) or matrix factorization
   - Learned factors: H (height), W (width), F (features), Core tensor

## Data Format

**TaxiBJ Flow Data:**
- Shape: `[n_timeslots, 2, 32, 32]`
- Channels: [inflow, outflow]
- Grid: 32×32 Beijing city representation
- Interval: 30 minutes (48 slots/day)

**External Features (per timeslot):**
- Temperature (continuous, normalized)
- Wind speed (continuous, normalized)
- Weather (17 categories, one-hot): Sunny, Cloudy, Rainy, Snowy, etc.
- Weekend flag (binary)
- Holiday flag (binary)
- **Total: 21 features**

## Research Questions

1. Does incorporating weather + calendar data improve prediction accuracy vs. baseline DeepLGR?
2. Which external factors contribute most (temperature, rain, holidays)?
3. Does external context improve robustness under abnormal conditions?

## Evaluation Metrics

- **MAE**: Mean Absolute Error (lower is better)
- **SMAPE**: Symmetric Mean Absolute Percentage Error (lower is better)

Both computed for:
- Overall performance
- Inflow vs Outflow separately
- Subgroups (weekday/weekend, holiday/regular, rainy/sunny)

## Configuration Options

### Temporal Parameters
- `--len_closeness`: Number of recent timeslots (default: 12)
- `--len_period`: Number of daily samples (default: 3)
- `--len_trend`: Number of weekly samples (default: 3)

### Training Parameters
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--epochs`: Max epochs (default: 100)
- `--patience`: Early stopping patience (default: 10)
- `--device`: cuda or cpu (default: cuda)

### Data Parameters
- `--year`: TaxiBJ year: 13, 14, 15, or 16 (default: 16)
- `--data_dir`: Raw data directory (default: data)
- `--processed_dir`: Preprocessed data directory (default: data/processed)

## Expected Results

Based on original DeepLGR paper:
- Training time: ~4-6 hours on GPU for 100 epochs
- BJ16 dataset: 7220 timeslots → ~5776 train, ~722 val, ~722 test samples
- Expected MAE: ~10-20 (baseline without external features)

## Development Notes

**External Feature Integration (TODO):**

The preprocessing pipeline loads and prepares external features, but they are not yet integrated into the DeepLGR model architecture. To complete the research contribution:

1. Modify `DeepLGR.__init__()` to accept external feature channels
2. Update `DeepLGR.forward()` to concatenate or fuse external features with flow inputs
3. Options for integration:
   - Concatenate as additional input channels to first conv layer
   - Use separate processing branch and fuse with SE block output
   - Add as conditioning to GlobalNet pooling layers

See `.github/copilot-instructions.md` for detailed architecture guidance.

## References

- Original DeepLGR: https://github.com/yoshall/DeepLGR
- Paper: "Revisiting Convolutional Neural Networks for Citywide Crowd Flow Analytics"
- Dataset: TaxiBJ from Deep Spatio-Temporal Residual Networks (AAAI 2017)
