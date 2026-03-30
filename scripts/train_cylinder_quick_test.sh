#!/bin/bash
# train_cylinder_quick_test.sh
# Quick test script for KBS-mHC-FNO on Cylinder dataset
# This script runs a minimal test with fewer epochs for quick validation

set -e

# =============================================================================
# 参数说明
# =============================================================================
# 训练脚本train_cylinder_mhc.py的主要参数：
#
# 参数名              | 说明                               | 示例值
# ----------------|-----------------------------------|--------
# --gpu            | GPU编号                             | 0, 1, 2...
# --batch_size     | 批大小                             | 8, 16, 32
# --epochs         | 训练轮数                           | 10 (快速), 500 (完整)
# --lr             | 学习率                             | 1e-3, 5e-4
# --save_name      | 实验名称，用于保存文件               | "test_name"
# --use_mhc        | 是否启用mHC (0/1)                  | 0=禁用, 1=启用
# --mhc_mode       | mHC模式                             | "discrete", "continuous"
# --mhc_expansion_ratio  | 流形扩展倍数              | 2, 4, 8
# --mhc_kdb_bandwidth    | 核密度平衡带宽        | 0.1, 0.5, 1.0
#
# 如何修改参数：
#   直接修改下面的配置值即可，或在python命令中添加对应的参数
# =============================================================================

# Quick test configuration
GPU="0"
BATCH_SIZE=8
EPOCHS=10  # Reduced epochs for quick testing
LR=1e-3

# Output base directory
LOG_DIR="../logs/Cylinder_mHC/quick_test"
CKPT_DIR="../checkpoints/Cylinder_mHC/quick_test"
RESULTS_DIR="../results/Cylinder_mHC/quick_test"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$CKPT_DIR"
mkdir -p "$RESULTS_DIR"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Quick Test: KBS-mHC-FNO Cylinder${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Configuration:"
echo "  GPU: $GPU"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS (快速测试，完整实验使用500)"
echo "  Learning rate: $LR"
echo ""

# =============================================================================
# 测试1：基线模型（无mHC）
# 参数说明：
#   --use_mhc 0: 禁用mHC（标准FNO）
#   其他mHC参数在禁用时无效，但保留用于一致性
# =============================================================================
echo -e "${YELLOW}测试1：基线模型（无mHC）${NC}"
python train_cylinder_mhc.py \
    --gpu "$GPU" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --save_name "quick_test_baseline" \
    --use_mhc 0 \
    --mhc_mode "continuous" \
    --mhc_expansion_ratio 4 \
    --mhc_kdb_bandwidth 0.5 \
    2>&1 | tee "$LOG_DIR/quick_test_baseline.log"

echo -e "${GREEN}✓ 测试1完成${NC}"
echo ""

# =============================================================================
# 测试2：mHC模型（continuous模式）
# 参数说明：
#   --use_mhc 1: 启用mHC
#   --mhc_mode "continuous": 使用continuous模式（推荐，效率更高）
#   --mhc_expansion_ratio 4: 流形扩展倍数为4
#   --mhc_kdb_bandwidth 0.5: 中等平滑强度
# =============================================================================
echo -e "${YELLOW}测试2：mHC模型（continuous模式）${NC}"
python train_cylinder_mhc.py \
    --gpu "$GPU" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --save_name "quick_test_mhc_continuous" \
    --use_mhc 1 \
    --mhc_mode "continuous" \
    --mhc_expansion_ratio 4 \
    --mhc_kdb_bandwidth 0.5 \
    2>&1 | tee "$LOG_DIR/quick_test_mhc_continuous.log"

echo -e "${GREEN}✓ 测试2完成${NC}"
echo ""

# =============================================================================
# 测试3：mHC模型（discrete模式）
# 参数说明：
#   --use_mhc 1: 启用mHC
#   --mhc_mode "discrete": 使用discrete模式（传统Sinkhorn迭代）
#   --mhc_expansion_ratio 4: 流形扩展倍数为4
#   --mhc_kdb_bandwidth 0.5: 中等平滑强度
# =============================================================================
echo -e "${YELLOW}测试3：mHC模型（discrete模式）${NC}"
python train_cylinder_mhc.py \
    --gpu "$GPU" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --save_name "quick_test_mhc_discrete" \
    --use_mhc 1 \
    --mhc_mode "discrete" \
    --mhc_expansion_ratio 4 \
    --mhc_kdb_bandwidth 0.5 \
    2>&1 | tee "$LOG_DIR/quick_test_mhc_discrete.log"

echo -e "${GREEN}✓ 测试3完成${NC}"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}快速测试完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "快速测试结果已保存到:"
echo "  日志文件:   $LOG_DIR"
echo "  模型检查点: $CKPT_DIR"
echo "  评估结果:   $RESULTS_DIR"
echo ""
echo "运行完整实验: bash run_cylinder_experiments.sh"
