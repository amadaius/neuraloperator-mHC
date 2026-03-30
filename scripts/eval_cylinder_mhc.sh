#!/bin/bash
# eval_cylinder_mhc.sh
# Evaluation script for KBS-mHC-FNO on Cylinder dataset
# This script loads saved checkpoints and runs evaluation

set -e

# Configuration
GPU="0"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Evaluation: KBS-mHC-FNO Cylinder${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Checkpoint names to evaluate (add more as needed)
CHECKPOINTS=(
    "baseline_no_mhc"
    "with_mhc_continuous"
    "mhc_mode_discrete"
    "mhc_mode_continuous"
    "expansion_ratio_2"
    "expansion_ratio_4"
    "expansion_ratio_8"
    "kdb_bandwidth_0.1"
    "kdb_bandwidth_0.5"
    "kdb_bandwidth_1.0"
)

for ckpt_name in "${CHECKPOINTS[@]}"; do
    echo -e "${YELLOW}Evaluating: $ckpt_name${NC}"

    python train_cylinder_mhc.py \
        --gpu "$GPU" \
        --save_name "$ckpt_name" \
        --eval 1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $ckpt_name evaluation completed${NC}"
    else
        echo -e "${RED}✗ $ckpt_name evaluation failed (checkpoint may not exist)${NC}"
    fi
    echo ""
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Evaluation completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to: ../results/Cylinder_mHC/"
