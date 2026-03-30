#!/bin/bash
# run_cylinder_experiments.sh
# Batch experiment runner for KBS-mHC-FNO on Cylinder dataset
# This script runs multiple experiments for comprehensive performance comparison

# 检查并激活conda环境
if [ -z "$CONDA_DEFAULT_ENV" ] || [ -z "$CONDA_PREFIX" ] || [ "$CONDA_DEFAULT_ENV" != "FNO2" ]; then
    # 强制使用FNO2环境（neuraloperator-mHC专用）
    echo "正在激活FNO2环境（neuraloperator-mHC专用）..."

    # 尝试常见的conda路径
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        source ~/miniconda3/etc/profile.d/conda.sh
    elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
        source ~/anaconda3/etc/profile.d/conda.sh
    elif [ -f ~/miniconda/etc/profile.d/conda.sh ]; then
        source ~/miniconda/etc/profile.d/conda.sh
    else
        # 使用which查找conda
        CONDA_PATH=$(which conda 2>/dev/null)
        if [ -n "$CONDA_PATH" ]; then
            source $(dirname $(dirname $CONDA_PATH))/etc/profile.d/conda.sh
        fi
    fi

    conda activate FNO2
    echo "✓ FNO2环境已激活"
    echo ""
fi

set -e  # Exit on error

# =============================================================================
# 参数说明表格
# =============================================================================
# 函数run_experiment()接收5个参数，按顺序如下：
#
# 位置 | 参数名              | 类型    | 说明                               | 示例值
# -----|-------------------|---------|-----------------------------------|--------
#  $1  | exp_name         | 字符串  | 实验名称，用于保存文件               | "baseline_no_mhc"
#  $2  | use_mhc          | 整数    | 是否启用mHC (0=禁用, 1=启用)       | 0 或 1
#  $3  | mhc_mode         | 字符串  | mHC操作模式                         | "discrete" 或 "continuous"
#  $4  | expansion_ratio   | 整数    | 流形扩展倍数                        | 2, 4, 8
#  $5  | kdb_bandwidth    | 浮点数  | 核密度平衡带宽                       | 0.1, 0.5, 1.0
#
# 使用示例：
#   run_experiment "实验名" 1 "continuous" 4 0.5
#   解释: 运行名为"实验名"的实验，启用mHC，使用continuous模式，expansion_ratio=4，kdb_bandwidth=0.5
#
# 如何修改参数：
#   1. 在run_experiment()调用中，按顺序修改5个参数
#   2. 例如：将第2个参数从1改为0，可以禁用mHC
#   3. 例如：将第4个参数从4改为8，可以增大expansion_ratio
# =============================================================================


# =============================================================================
# 全局配置（所有实验共用的参数）
# =============================================================================
GPU="2"              # GPU编号
BATCH_SIZE=8          # 批大小
EPOCHS=500            # 训练轮数
LR=1e-3               # 学习率

# Output base directory
LOG_DIR="../logs/Cylinder_mHC"
CKPT_DIR="../checkpoints/Cylinder_mHC"
RESULTS_DIR="../results/Cylinder_mHC"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$CKPT_DIR"
mkdir -p "$RESULTS_DIR"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}KBS-mHC-FNO Cylinder Experiments${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 函数：运行单个实验
# 参数说明（按顺序）：
#   $1 - exp_name: 实验名称，用于保存文件
#   $2 - use_mhc: 是否启用mHC (0=禁用, 1=启用)
#   $3 - mhc_mode: mHC操作模式 ("discrete"或"continuous")
#   $4 - expansion_ratio: 流形扩展倍数 (2/4/8等)
#   $5 - kdb_bandwidth: 核密度平衡带宽 (0.1/0.5/1.0等)
#
# 使用方法示例：
#   run_experiment "实验名" 1 "continuous" 4 0.5
#   解释: 运行名为"实验名"的实验，启用mHC，使用continuous模式，expansion_ratio=4，kdb_bandwidth=0.5
run_experiment() {
    local exp_name=$1            # 第1个参数：实验名称
    local use_mhc=$2             # 第2个参数：是否启用mHC (0/1)
    local mhc_mode=$3            # 第3个参数：mHC模式 ("discrete"/"continuous")
    local expansion_ratio=$4       # 第4个参数：流形扩展倍数
    local kdb_bandwidth=$5       # 第5个参数：核密度平衡带宽

    echo -e "${YELLOW}Running experiment: $exp_name${NC}"
    echo "  实验名称: $exp_name"
    echo "  是否启用mHC: $use_mhc (0=禁用, 1=启用)"
    echo "  mHC模式: $mhc_mode (discrete/continuous)"
    echo "  流形扩展倍数: $expansion_ratio (控制流形表达能力，越大越强但参数更多)"
    echo "  核密度平衡带宽: $kdb_bandwidth (控制空间平滑，0.1=弱, 0.5=中, 1.0=强)"
    echo ""

    python train_cylinder_mhc.py \
        --gpu "$GPU" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --save_name "$exp_name" \
        --use_mhc "$use_mhc" \
        --mhc_mode "$mhc_mode" \
        --mhc_expansion_ratio "$expansion_ratio" \
        --mhc_kdb_bandwidth "$kdb_bandwidth" \
        2>&1 | tee "$LOG_DIR/${exp_name}.log"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $name completed successfully${NC}"
    else
        echo -e "${RED}✗ $name failed${NC}"
        return 1
    fi
    echo ""
}

# ==============================================================================
# 实验1：有无mHC对比
# 目的：对比标准FNO和mHC-FNO的性能差异
# ==============================================================================
echo -e "${YELLOW}=== 实验1：有无mHC对比 ===${NC}"
echo ""

# 实验1.1：基线模型（无mHC）
# 参数解释：
#   - "baseline_no_mhc": 实验名称
#   - 0: 不启用mHC（标准FNO）
#   - "continuous": mHC模式（无mHC时此参数无效）
#   - 4: expansion_ratio（无mHC时此参数无效）
#   - 0.5: kdb_bandwidth（无mHC时此参数无效）
run_experiment "baseline_no_mhc" 0 "continuous" 4 0.5

# 实验1.2：mHC模型（continuous模式）
# 参数解释：
#   - "with_mhc_continuous": 实验名称
#   - 1: 启用mHC
#   - "continuous": 使用continuous模式（推荐，效率更高）
#   - 4: expansion_ratio=4（中等流形扩展）
#   - 0.5: kdb_bandwidth=0.5（中等平滑）
run_experiment "with_mhc_continuous" 1 "continuous" 4 0.5

# ==============================================================================
# 实验2：mHC模式对比（discrete vs continuous）
# 目的：对比两种mHC操作模式的性能差异
# ==============================================================================
echo -e "${YELLOW}=== 实验2：mHC模式对比 ===${NC}"
echo ""

# 实验2.1：discrete模式
# 参数解释：
#   - "mhc_mode_discrete": 实验名称
#   - 1: 启用mHC
#   - "discrete": 使用discrete模式（传统通道级Sinkhorn）
#   - 4: expansion_ratio=4
#   - 0.5: kdb_bandwidth=0.5
run_experiment "mhc_mode_discrete" 1 "discrete" 4 0.5

# 实验2.2：continuous模式
# 参数解释：
#   - "mhc_mode_continuous": 实验名称
#   - 1: 启用mHC
#   - "continuous": 使用continuous模式（推荐，使用KDB）
#   - 4: expansion_ratio=4
#   - 0.5: kdb_bandwidth=0.5
run_experiment "mhc_mode_continuous" 1 "continuous" 4 0.5

# ==============================================================================
# 实验3：expansion_ratio对比（流形扩展倍数）
# 目的：对比不同流形扩展倍数对性能的影响
# 说明：expansion_ratio控制mHC流形的扩展倍数，值越大表达能力越强但参数更多
# ==============================================================================
echo -e "${YELLOW}=== 实验3：expansion_ratio对比 ===${NC}"
echo ""

# 实验3.1：expansion_ratio=2（小流形，参数少）
# 参数解释：
#   - "expansion_ratio_2": 实验名称
#   - 1: 启用mHC
#   - "continuous": 使用continuous模式
#   - 2: expansion_ratio=2（较小的流形维度）
#   - 0.5: kdb_bandwidth=0.5
run_experiment "expansion_ratio_2" 1 "continuous" 2 0.5

# 实验3.2：expansion_ratio=4（中等流形，推荐）
# 参数解释：
#   - "expansion_ratio_4": 实验名称
#   - 1: 启用mHC
#   - "continuous": 使用continuous模式
#   - 4: expansion_ratio=4（中等流形维度，平衡性能）
#   - 0.5: kdb_bandwidth=0.5
run_experiment "expansion_ratio_4" 1 "continuous" 4 0.5

# 实验3.3：expansion_ratio=8（大流形，参数多）
# 参数解释：
#   - "expansion_ratio_8": 实验名称
#   - 1: 启用mHC
#   - "continuous": 使用continuous模式
#   - 8: expansion_ratio=8（较大的流形维度，表达能力强）
#   - 0.5: kdb_bandwidth=0.5
run_experiment "expansion_ratio_8" 1 "continuous" 8 0.5

# ==============================================================================
# 实验4：kdb_bandwidth对比（核密度平衡带宽）
# 目的：对比不同核密度平衡带宽对性能的影响
# 说明：kdb_bandwidth控制空间平滑程度，值越小保留细节越多，值越大梯度越稳定
# ==============================================================================
echo -e "${YELLOW}=== 实验4：kdb_bandwidth对比 ===${NC}"
echo ""

# 实验4.1：kdb_bandwidth=0.1（弱平滑，保留细节）
# 参数解释：
#   - "kdb_bandwidth_0.1": 实验名称
#   - 1: 启用mHC
#   - "continuous": 使用continuous模式
#   - 4: expansion_ratio=4
#   - 0.1: kdb_bandwidth=0.1（弱平滑，保留更多局部细节）
run_experiment "kdb_bandwidth_0.1" 1 "continuous" 4 0.1

# 实验4.2：kdb_bandwidth=0.5（中等平滑，推荐）
# 参数解释：
#   - "kdb_bandwidth_0.5": 实验名称
#   - 1: 启用mHC
#   - "continuous": 使用continuous模式
#   - 4: expansion_ratio=4
#   - 0.5: kdb_bandwidth=0.5（中等平滑，平衡稳定性和细节保留）
run_experiment "kdb_bandwidth_0.5" 1 "continuous" 4 0.5

# 实验4.3：kdb_bandwidth=1.0（强平滑，稳定）
# 参数解释：
#   - "kdb_bandwidth_1.0": 实验名称
#   - 1: 启用mHC
#   - "continuous": 使用continuous模式
#   - 4: expansion_ratio=4
#   - 1.0: kdb_bandwidth=1.0（强平滑，梯度更稳定）
run_experiment "kdb_bandwidth_1.0" 1 "continuous" 4 1.0

# ==============================================================================
# SUMMARY
# ==============================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All experiments completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to:"
echo "  Logs:      $LOG_DIR"
echo "  Checkpoints: $CKPT_DIR"
echo "  Results:   $RESULTS_DIR"
echo ""
