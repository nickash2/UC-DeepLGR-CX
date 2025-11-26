#!/bin/bash
# filepath: /workspaces/project/train_all_periods.sh
# Train and evaluate separate models for each period (P1, P2, P3, P4)
# Skips periods that already have results

set -e  # Exit on error

# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1

# Configuration
PERIODS=("P1" "P2" "P3" "P4")
LR=0.0001
BATCH_SIZE=16
EPOCHS=200
PATIENCE=10
DEVICE="cuda"
NUM_WORKERS=4

echo "========================================================================"
echo "TRAIN AND EVALUATE SEPARATE MODELS FOR EACH PERIOD"
echo "========================================================================"
echo "Periods: ${PERIODS[@]}"
echo "Learning rate: $LR"
echo "Batch size: $BATCH_SIZE"
echo "Max epochs: $EPOCHS"
echo "CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "========================================================================"
echo ""

# Create results directory
mkdir -p results_per_period

# Initialize CSV file if it doesn't exist
CSV_FILE="results_per_period/all_periods_results.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "Period,Model,MAE,SMAPE,MAE_Inflow,MAE_Outflow,Train_Samples,Val_Samples,Test_Samples,Best_Epoch,Val_Loss" > $CSV_FILE
fi

# Function to clear CUDA cache
clear_cuda_cache() {
    echo "Clearing CUDA cache..."
    python -c "import torch; torch.cuda.empty_cache(); print(f'GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB')"
}

# ========================================================================
# Train and Evaluate Each Period
# ========================================================================

for PERIOD in "${PERIODS[@]}"; do
    
    # Clear CUDA cache before each period
    clear_cuda_cache
    
    # Create checkpoint directories under common directory
    CHECKPOINT_DIR_BASELINE="checkpoints_per_period/${PERIOD}_baseline"
    CHECKPOINT_DIR_EXTENDED="checkpoints_per_period/${PERIOD}_extended"
    
    # Check if both baseline and extended are already done
    BASELINE_DONE=false
    EXTENDED_DONE=false
    
    if [ -f "$CHECKPOINT_DIR_BASELINE/best.pth" ] && [ -f "results_per_period/${PERIOD}_baseline/evaluation_results.json" ]; then
        BASELINE_DONE=true
    fi
    
    if [ -f "$CHECKPOINT_DIR_EXTENDED/best.pth" ] && [ -f "results_per_period/${PERIOD}_extended/evaluation_results.json" ]; then
        EXTENDED_DONE=true
    fi
    
    if [ "$BASELINE_DONE" = true ] && [ "$EXTENDED_DONE" = true ]; then
        echo "========================================================================"
        echo "SKIPPING PERIOD: $PERIOD (already completed)"
        echo "========================================================================"
        echo ""
        continue
    fi
    
    echo "========================================================================"
    echo "PROCESSING PERIOD: $PERIOD"
    echo "========================================================================"
    echo ""
    
    mkdir -p $CHECKPOINT_DIR_BASELINE
    mkdir -p $CHECKPOINT_DIR_EXTENDED
    
    # ====================================================================
    # Train Baseline Model
    # ====================================================================
    if [ "$BASELINE_DONE" = false ]; then
        echo "--------------------------------------------------------------------"
        echo "Training Baseline Model on $PERIOD"
        echo "--------------------------------------------------------------------"
        echo ""
        
        # Clear cache before training
        clear_cuda_cache
        
        # Use set +e temporarily to not exit on error
        set +e
        python src/train.py \
            --data_dir data/processed \
            --year $PERIOD \
            --checkpoint_dir $CHECKPOINT_DIR_BASELINE \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --epochs $EPOCHS \
            --patience $PATIENCE \
            --device $DEVICE \
            --num_workers $NUM_WORKERS \
            --len_closeness 5 \
            --len_period 3 \
            --len_trend 3
        
        TRAIN_EXIT_CODE=$?
        set -e
        
        if [ $TRAIN_EXIT_CODE -ne 0 ]; then
            echo ""
            echo "ERROR: Baseline training failed for $PERIOD with exit code $TRAIN_EXIT_CODE"
            echo "Attempting to reset CUDA and continue..."
            clear_cuda_cache
            sleep 5
            continue
        fi
        
        echo ""
        echo "Baseline training complete for $PERIOD"
        echo ""
        
        # ====================================================================
        # Evaluate Baseline Model
        # ====================================================================
        echo "--------------------------------------------------------------------"
        echo "Evaluating Baseline Model on $PERIOD"
        echo "--------------------------------------------------------------------"
        echo ""
        
        python src/evaluate.py \
            --data_dir data/processed \
            --year $PERIOD \
            --checkpoint $CHECKPOINT_DIR_BASELINE/best.pth \
            --output_dir results_per_period/${PERIOD}_baseline \
            --batch_size $BATCH_SIZE \
            --device $DEVICE \
            --num_workers $NUM_WORKERS \
            --len_closeness 5 \
            --len_period 3 \
            --len_trend 3
        
        echo ""
        echo "Baseline evaluation complete for $PERIOD"
        echo ""
        
        # Extract baseline results and append to CSV
        if [ -f "results_per_period/${PERIOD}_baseline/evaluation_results.json" ]; then
            python -c "
import json
with open('results_per_period/${PERIOD}_baseline/evaluation_results.json', 'r') as f:
    data = json.load(f)
    results = data['overall']
with open('$CHECKPOINT_DIR_BASELINE/best.pth', 'rb') as f:
    import torch
    ckpt = torch.load(f, map_location='cpu')
# Get test sample count from subgroups
n_test = data['subgroups']['regular']['count']
print(f\"$PERIOD,Baseline,{results['mae_overall']:.4f},{results['smape_overall']:.2f},{results['mae_inflow']:.4f},{results['mae_outflow']:.4f},N/A,N/A,{n_test},{ckpt['epoch']},{ckpt['val_loss']:.6f}\")
" >> $CSV_FILE
        fi
    else
        echo "Skipping baseline training for $PERIOD (already done)"
        echo ""
    fi
    
    # ====================================================================
    # Train Extended Model
    # ====================================================================
    if [ "$EXTENDED_DONE" = false ]; then
        echo "--------------------------------------------------------------------"
        echo "Training Extended Model (with external features) on $PERIOD"
        echo "--------------------------------------------------------------------"
        echo ""
        
        # Clear cache before training
        clear_cuda_cache
        
        # Use set +e temporarily to not exit on error
        set +e
        python src/train.py \
            --data_dir data/processed \
            --year $PERIOD \
            --checkpoint_dir $CHECKPOINT_DIR_EXTENDED \
            --batch_size $BATCH_SIZE \
            --lr $LR \
            --epochs $EPOCHS \
            --patience $PATIENCE \
            --device $DEVICE \
            --num_workers $NUM_WORKERS \
            --len_closeness 5 \
            --len_period 3 \
            --len_trend 3 \
            --use_external
        
        TRAIN_EXIT_CODE=$?
        set -e
        
        if [ $TRAIN_EXIT_CODE -ne 0 ]; then
            echo ""
            echo "ERROR: Extended training failed for $PERIOD with exit code $TRAIN_EXIT_CODE"
            echo "Attempting to reset CUDA and continue..."
            clear_cuda_cache
            sleep 5
            continue
        fi
        
        echo ""
        echo "Extended training complete for $PERIOD"
        echo ""
        
        # ====================================================================
        # Evaluate Extended Model
        # ====================================================================
        echo "--------------------------------------------------------------------"
        echo "Evaluating Extended Model on $PERIOD"
        echo "--------------------------------------------------------------------"
        echo ""
        
        python src/evaluate.py \
            --data_dir data/processed \
            --year $PERIOD \
            --checkpoint $CHECKPOINT_DIR_EXTENDED/best.pth \
            --output_dir results_per_period/${PERIOD}_extended \
            --batch_size $BATCH_SIZE \
            --device $DEVICE \
            --num_workers $NUM_WORKERS \
            --len_closeness 5 \
            --len_period 3 \
            --len_trend 3 \
            --use_external
        
        echo ""
        echo "Extended evaluation complete for $PERIOD"
        echo ""
        
        # Extract extended results and append to CSV
        if [ -f "results_per_period/${PERIOD}_extended/evaluation_results.json" ]; then
            python -c "
import json
with open('results_per_period/${PERIOD}_extended/evaluation_results.json', 'r') as f:
    data = json.load(f)
    results = data['overall']
with open('$CHECKPOINT_DIR_EXTENDED/best.pth', 'rb') as f:
    import torch
    ckpt = torch.load(f, map_location='cpu')
# Get test sample count from subgroups
n_test = data['subgroups']['regular']['count']
print(f\"$PERIOD,Extended,{results['mae_overall']:.4f},{results['smape_overall']:.2f},{results['mae_inflow']:.4f},{results['mae_outflow']:.4f},N/A,N/A,{n_test},{ckpt['epoch']},{ckpt['val_loss']:.6f}\")
" >> $CSV_FILE
        fi
    else
        echo "Skipping extended training for $PERIOD (already done)"
        echo ""
    fi
    
    echo "========================================================================"
    echo "$PERIOD COMPLETE"
    echo "========================================================================"
    echo ""
done

# ========================================================================
# Summary
# ========================================================================
echo "========================================================================"
echo "ALL PERIODS COMPLETE - SUMMARY"
echo "========================================================================"
echo ""

echo "Results CSV: $CSV_FILE"
echo ""
echo "Results:"
cat $CSV_FILE
echo ""

echo "Average Performance Across Periods:"
python -c "
import csv
with open('$CSV_FILE', 'r') as f:
    reader = csv.DictReader(f)
    baseline_maes = []
    extended_maes = []
    for row in reader:
        if row['Model'] == 'Baseline':
            baseline_maes.append(float(row['MAE']))
        elif row['Model'] == 'Extended':
            extended_maes.append(float(row['MAE']))
    
    if baseline_maes:
        print(f'  Baseline MAE: {sum(baseline_maes)/len(baseline_maes):.4f}')
    if extended_maes:
        print(f'  Extended MAE: {sum(extended_maes)/len(extended_maes):.4f}')
    if baseline_maes and extended_maes:
        avg_baseline = sum(baseline_maes)/len(baseline_maes)
        avg_extended = sum(extended_maes)/len(extended_maes)
        improvement = ((avg_baseline - avg_extended) / avg_baseline) * 100
        print(f'  Improvement: {improvement:.2f}%' if improvement > 0 else f'  Degradation: {-improvement:.2f}%')
"

echo ""
echo "========================================================================"
echo "PIPELINE COMPLETE!"
echo "========================================================================"