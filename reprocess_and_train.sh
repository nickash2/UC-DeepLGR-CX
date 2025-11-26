#!/bin/bash
# Reprocess all period data with proper seeds, then train all models
# This ensures full reproducibility

set -e  # Exit on error

echo "========================================================================"
echo "FULL REPRODUCIBLE PIPELINE: REPROCESS + TRAIN"
echo "========================================================================"
echo ""

# ========================================================================
# Step 1: Ask about backup
# ========================================================================
BACKUP_DATA=false
BACKUP_MODELS=false

if [ -d "data/processed" ] && ls data/processed/BJP*.npz 1> /dev/null 2>&1; then
    echo "Existing processed data found."
    read -p "Do you want to backup processed data? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        BACKUP_DATA=true
    fi
fi

if [ -d "checkpoints_per_period" ] || [ -d "results_per_period" ]; then
    echo "Existing checkpoints/results found."
    read -p "Do you want to backup checkpoints and results? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        BACKUP_MODELS=true
    fi
fi
echo ""

# ========================================================================
# Step 2: Backup old data (if requested)
# ========================================================================
if [ "$BACKUP_DATA" = true ]; then
    echo "Backing up old processed data..."
    mkdir -p backups
    BACKUP_DIR="backups/processed_$(date +%Y%m%d_%H%M%S)"
    echo "Creating backup: $BACKUP_DIR"
    cp -r data/processed $BACKUP_DIR
    echo "Backup complete!"
    echo ""
fi

# ========================================================================
# Step 3: Delete old processed data
# ========================================================================
echo "Removing old processed data..."
rm -rf data/processed/BJP*.npz data/processed/BJP*_stats.pkl
echo "Old processed data removed."
echo ""

# ========================================================================
# Step 4: Reprocess all periods with seed=42
# ========================================================================
echo "Reprocessing all periods (P1, P2, P3, P4) with seed=42..."
python src/preprocess_periods.py \
    --periods all \
    --data_dir data \
    --output_dir data/processed \
    --len_closeness 5 \
    --len_period 3 \
    --len_trend 3 \
    --seed 42

echo ""
echo "Preprocessing complete!"
echo ""

# ========================================================================
# Step 5: Backup old checkpoints and results (if requested)
# ========================================================================
if [ "$BACKUP_MODELS" = true ]; then
    echo "Backing up old checkpoints and results..."
    mkdir -p backups
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    if [ -d "checkpoints_per_period" ]; then
        CKPT_BACKUP="backups/checkpoints_${TIMESTAMP}"
        echo "Creating checkpoint backup: $CKPT_BACKUP"
        cp -r checkpoints_per_period $CKPT_BACKUP
    fi

    if [ -d "results_per_period" ]; then
        RESULTS_BACKUP="backups/results_${TIMESTAMP}"
        echo "Creating results backup: $RESULTS_BACKUP"
        cp -r results_per_period $RESULTS_BACKUP
    fi
    echo "Backups complete!"
    echo ""
fi

# ========================================================================
# Step 6: Clean old training artifacts
# ========================================================================
echo "Removing old training checkpoints and results..."
rm -rf checkpoints_per_period
rm -rf results_per_period
echo "Old training artifacts removed."
echo ""

# ========================================================================
# Step 7: Train all models with seeds
# ========================================================================
echo "Training all models (Baseline + Extended) for all periods..."
echo ""
bash train_all_periods.sh

echo ""
echo "========================================================================"
echo "FULL PIPELINE COMPLETE!"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  - All data reprocessed with seed=42"
echo "  - All models trained with seed=42"
echo "  - Results saved to: results_per_period/all_periods_results.csv"
echo ""
if [ "$BACKUP_DATA" = true ] || [ "$BACKUP_MODELS" = true ]; then
    echo "Backups saved to:"
    if [ "$BACKUP_DATA" = true ] && [ -n "$BACKUP_DIR" ]; then
        echo "  - Data: $BACKUP_DIR"
    fi
    if [ "$BACKUP_MODELS" = true ]; then
        if [ -n "$CKPT_BACKUP" ]; then
            echo "  - Checkpoints: $CKPT_BACKUP"
        fi
        if [ -n "$RESULTS_BACKUP" ]; then
            echo "  - Results: $RESULTS_BACKUP"
        fi
    fi
    echo ""
fi
echo "Ready for reproducible experiments!"
echo "========================================================================"
