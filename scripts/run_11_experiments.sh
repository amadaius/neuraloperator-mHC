#!/bin/bash
# Batch run 11 mHC experiments on Cylinder dataset

source ~/miniconda3/etc/profile.d/conda.sh
conda activate FNO2

cd /home/leeshu/wmm/neuraloperator-mHC/scripts

# Common parameters
GPU_LIST=(0 1 2)
BATCH_SIZE=10
EPOCHS=500
LR=1e-3

# Experiment configurations
declare -a EXP_CONFIGS=(
    # Exp 1: Baseline (no mHC) - already done as Phase 1
    "--use_mhc 0 --save_name exp1_baseline_no_mhc"
    
    # Exp 2-3: mHC mode comparison
    "--use_mhc 1 --mhc_mode discrete --save_name exp2_mhc_mode_discrete"
    "--use_mhc 1 --mhc_mode continuous --save_name exp3_mhc_mode_continuous"
    
    # Exp 4-6: Expansion ratio comparison
    "--use_mhc 1 --mhc_expansion_ratio 2 --save_name exp4_expansion_ratio_2"
    "--use_mhc 1 --mhc_expansion_ratio 4 --save_name exp5_expansion_ratio_4"
    "--use_mhc 1 --mhc_expansion_ratio 8 --save_name exp6_expansion_ratio_8"
    
    # Exp 7-9: KDB bandwidth comparison
    "--use_mhc 1 --mhc_kdb_bandwidth 0.1 --save_name exp7_kdb_bandwidth_0.1"
    "--use_mhc 1 --mhc_kdb_bandwidth 0.5 --save_name exp8_kdb_bandwidth_0.5"
    "--use_mhc 1 --mhc_kdb_bandwidth 1.0 --save_name exp9_kdb_bandwidth_1.0"
    
    # Exp 10: Best combination (continuous mode, ratio=4, bandwidth=0.5)
    "--use_mhc 1 --mhc_mode continuous --mhc_expansion_ratio 4 --mhc_kdb_bandwidth 0.5 --save_name exp10_best_mhc"
    
    # Exp 11: Pushforward with best mHC
    "--use_mhc 1 --mhc_mode continuous --mhc_expansion_ratio 4 --mhc_kdb_bandwidth 0.5 --use_pushforward 1 --warmup_epochs 100 --max_rollout_steps 5 --temporal_gamma 0.8 --save_name exp11_pushforward_best_mhc"
)

# Function to run single experiment
run_experiment() {
    local gpu_id=$1
    local config=$2
    local exp_num=$3
    
    echo "Starting Experiment $exp_num on GPU $gpu_id..."
    
    nohup python train_cylinder_mhc.py \
        --gpu $gpu_id \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --lr $LR \
        $config \
        > /tmp/exp_${exp_num}.log 2>&1 &
    
    local pid=$!
    echo "Experiment $exp_num started with PID: $pid"
    return $pid
}

# Run experiments sequentially on available GPUs
EXP_COUNT=11
GPU_COUNT=3

for ((i=0; i<EXP_COUNT; i++)); do
    gpu_id=${GPU_LIST[$((i % GPU_COUNT))]}
    
    config="${EXP_CONFIGS[$i]}"
    exp_num=$((i+1))
    
    run_experiment $gpu_id "$config" $exp_num
    
    # Wait a bit between launches to avoid resource spikes
    sleep 5
done

echo "All 11 experiments launched!"
echo "Monitor logs with: tail -f /tmp/exp_*.log"
