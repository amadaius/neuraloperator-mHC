# KBS-mHC-FNO Cylinder实验完整指南

本目录包含用于Cylinder数据集的KBS-mHC-FNO训练、评估和监控的完整脚本集。

## 📁 文件说明

### 核心训练脚本

- **train_cylinder_mhc.py**: 主训练脚本（32KB）
  - 支持完整的KBS-mHC-FNO架构
  - 支持有无mHC的对比
  - 支持discrete和continuous两种mHC模式
  - 支持自定义mHC参数（expansion_ratio, kdb_bandwidth等）
  - 自动保存日志、检查点和评估结果
  - 使用滑窗数据集进行单步训练
  - 支持autoregressive评估

### 辅助脚本

- **train_cylinder_quick_test.sh**: 快速测试脚本（5.1KB）
  - 运行3个基础测试（每个10 epochs）
  - 用于验证脚本和环境的正确性
  - 输出到`logs/Cylinder_mHC/quick_test/`

- **run_cylinder_experiments.sh**: 完整实验脚本（11KB）
  - 运行所有11个性能对比实验（每个500 epochs）
  - 包括4个维度的对比实验
  - 自动激活FNO2环境（neuraloperator-mHC专用）
  - 顺序执行，一个接一个运行

- **eval_cylinder_mhc.sh**: 评估脚本（1.4KB）
  - 批量加载保存的检查点
  - 运行评估并生成详细报告
  - 生成可视化结果

- **monitor_experiments.sh**: 实时监控脚本（7.0KB）
  - 实时显示所有实验的运行状态
  - 显示当前epoch、loss、GPU使用率
  - 支持5秒自动刷新

---

## 🚀 快速开始

### 环境要求

- **Conda环境**: FNO2（neuraloperator-mHC专用）
- **GPU**: 建议1-3个GPU
- **CUDA**: 12.x
- **PyTorch**: 2.5.1+

### 步骤1：激活FNO2环境

```bash
conda activate FNO2
```

**验证环境**：
```bash
python -c "
from neuralop.models import FNO
import inspect
sig = inspect.signature(FNO.__init__)
print('✓ FNO支持use_mhc:', 'use_mhc' in sig.parameters)
print('✓ FNO支持mhc_mode:', 'mhc_mode' in sig.parameters)
print('✓ neuralop路径:', FNO.__module__)
"
```

应该显示：`✓ FNO支持use_mhc: True`

### 步骤2：运行快速测试（推荐首次使用）

```bash
cd /home/leeshu/wmm/neuraloperator-mHC/scripts
bash train_cylinder_quick_test.sh
```

这将运行3个测试来验证环境配置：
1. baseline_no_mhc - 无mHC的基线模型
2. mhc_continuous - continuous模式的mHC
3. mhc_discrete - discrete模式的mHC

### 步骤3：运行完整实验

```bash
cd /home/leeshu/wmm/neuraloperator-mHC/scripts
bash run_cylinder_experiments.sh
```

这将运行所有11个对比实验（详见下方实验列表）。

### 步骤4：监控实验进度（可选，另一个终端）

```bash
conda activate FNO2
cd /home/leeshu/wmm/neuraloperator-mHC/scripts
bash monitor_experiments.sh
```

---

## 📊 实验列表（11个）

### 实验1：有无mHC对比（2个实验）

| 实验名称 | use_mhc | mhc_mode | expansion_ratio | kdb_bandwidth | 说明 |
|---------|---------|-----------|-----------------|----------------|------|
| baseline_no_mhc | 0 | continuous | 4 | 0.5 | 基线模型（标准FNO） |
| with_mhc_continuous | 1 | continuous | 4 | 0.5 | mHC模型（continuous模式） |

**对比维度**: 有无mHC的性能差异

### 实验2：mHC模式对比（2个实验）

| 实验名称 | use_mhc | mhc_mode | expansion_ratio | kdb_bandwidth | 说明 |
|---------|---------|-----------|-----------------|----------------|------|
| mhc_mode_discrete | 1 | discrete | 4 | 0.5 | discrete模式（传统Sinkhorn） |
| mhc_mode_continuous | 1 | continuous | 4 | 0.5 | continuous模式（KDB，推荐） |

**对比维度**: discrete vs continuous模式性能差异

### 实验3：expansion_ratio对比（3个实验）

| 实验名称 | use_mhc | mhc_mode | expansion_ratio | kdb_bandwidth | 说明 |
|---------|---------|-----------|-----------------|----------------|------|
| expansion_ratio_2 | 1 | continuous | 2 | 0.5 | 小流形扩展（64维） |
| expansion_ratio_4 | 1 | continuous | 4 | 0.5 | 中等流形扩展（128维），推荐 |
| expansion_ratio_8 | 1 | continuous | 8 | 0.5 | 大流形扩展（256维） |

**对比维度**: 不同流形扩展倍数的影响

**计算公式**: 流形维度 = expansion_ratio × hidden_channels (32)

### 实验4：kdb_bandwidth对比（3个实验）

| 实验名称 | use_mhc | mhc_mode | expansion_ratio | kdb_bandwidth | 说明 |
|---------|---------|-----------|-----------------|----------------|------|
| kdb_bandwidth_0.1 | 1 | continuous | 4 | 0.1 | 弱平滑，保留细节 |
| kdb_bandwidth_0.5 | 1 | continuous | 4 | 0.5 | 中等平滑，推荐 |
| kdb_bandwidth_1.0 | 1 | continuous | 4 | 1.0 | 强平滑，梯度稳定 |

**对比维度**: 不同核密度平衡带宽的影响

---

## 🔧 参数说明

### Shell脚本函数参数

`run_experiment()` 函数接收5个位置参数：

| 位置 | 参数 | 类型 | 说明 | 示例值 |
|------|------|------|------|--------|
| $1 | exp_name | 字符串 | 实验名称 | `"baseline_no_mhc"` |
| $2 | use_mhc | 整数 | 是否启用mHC | `0`或`1` |
| $3 | mhc_mode | 字符串 | mHC操作模式 | `"discrete"`或`"continuous"` |
| $4 | expansion_ratio | 整数 | 流形扩展倍数 | `2`, `4`, `8` |
| $5 | kdb_bandwidth | 浮点数 | 核密度平衡带宽 | `0.1`, `0.5`, `1.0` |

**调用示例**：
```bash
run_experiment "my_experiment" 1 "continuous" 4 0.5
```

### Python训练脚本主要参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--gpu` | GPU编号 | 0 | 0, 1, 2 |
| `--batch_size` | 批大小 | 8 | 4, 8, 16 |
| `--epochs` | 训练轮数 | 500 | 10(测试), 500(完整) |
| `--lr` | 学习率 | 1e-3 | 5e-4, 1e-3 |
| `--save_name` | 实验名称 | - | 必填 |
| `--use_mhc` | 是否启用mHC | 1 | 1(启用) |
| `--mhc_mode` | mHC模式 | continuous | continuous |
| `--mhc_expansion_ratio` | 流形扩展倍数 | 4 | 4 |
| `--mhc_kdb_bandwidth` | 核密度平衡带宽 | 0.5 | 0.5 |
| `--modes` | FNO傅里叶模式数 | 12 | 8, 12, 16 |
| `--width` | 隐藏通道数 | 32 | 16, 32, 64 |
| `--n_layers` | FNO层数 | 4 | 2, 4, 6 |

### mHC参数详细说明

#### use_mhc（是否启用mHC）
- `0`: 禁用mHC，使用标准FNO
- `1`: 启用mHC，使用mHC-FNO（推荐）

#### mhc_mode（mHC操作模式）
- `"discrete"`: 传统通道级Sinkhorn迭代，复杂度O(N·n²)
- `"continuous"`: 连续KDB模式，使用深度卷积，复杂度O(N·K²)，**推荐**

#### expansion_ratio（流形扩展倍数）
- 控制mHC流形的扩展倍数：流形维度 = expansion_ratio × hidden_channels
- `2`: 小扩展，参数少（64维）
- `4`: 中等扩展，平衡性能和效率（128维），**推荐**
- `8`: 大扩展，表达能力强（256维）

#### kdb_bandwidth（核密度平衡带宽）
- 控制空间平滑程度（仅对continuous模式有效）
- `0.1`: 弱平滑，保留更多局部细节
- `0.5`: 中等平滑，平衡稳定性和细节保留，**推荐**
- `1.0`: 强平滑，梯度更稳定

---

## 💻 使用示例

### 运行单个实验

```bash
conda activate FNO2
cd /home/leeshu/wmm/neuraloperator-mHC/scripts

python train_cylinder_mhc.py \
    --gpu 2 \
    --batch_size 8 \
    --epochs 500 \
    --save_name "my_experiment" \
    --use_mhc 1 \
    --mhc_mode "continuous" \
    --mhc_expansion_ratio 4 \
    --mhc_kdb_bandwidth 0.5
```

### 修改实验参数

编辑`run_cylinder_experiments.sh`，找到对应的实验调用：

```bash
# 原始（第122行）
run_experiment "baseline_no_mhc" 0 "continuous" 4 0.5

# 修改为：禁用mHC，但使用更大的模型
run_experiment "baseline_no_mhc" 0 "continuous" 8 0.5
#                              ↑ 修改这里：expansion_ratio从4改为8
```

---

## 📂 输出目录结构

```
logs/Cylinder_mHC/
├── quick_test/              # 快速测试日志
│   ├── quick_test_baseline.log
│   ├── quick_test_mhc_continuous.log
│   └── quick_test_mhc_discrete.log
├── baseline_no_mhc.log      # 完整实验日志
├── with_mhc_continuous.log
├── mhc_mode_discrete.log
└── ... (共11个日志文件)

checkpoints/Cylinder_mHC/
├── quick_test/             # 快速测试检查点
│   ├── quick_test_baseline.pt
│   ├── quick_test_mhc_continuous.pt
│   └── quick_test_mhc_discrete.pt
├── baseline_no_mhc.pt      # 完整实验检查点
├── with_mhc_continuous.pt
└── ... (共11个检查点文件)

results/Cylinder_mHC/
├── quick_test/             # 快速测试结果
│   ├── quick_test_baseline_results.mat
│   ├── quick_test_baseline_metrics.csv
│   └── ...
├── baseline_no_mhc_results.mat  # 完整实验结果
├── baseline_no_mhc_metrics.csv
└── ... (共11个实验结果)
```

---

## 📈 评估指标

训练和评估使用以下指标：

1. **Relative L2 Error**: 相对L2误差（主要指标）
2. **MAE**: 平均绝对误差
3. **RMSE**: 均方根误差
4. **Max AE**: 最大绝对误差

所有指标都会：
- 保存每时间步的详细结果
- 计算总体平均值
- 支持mask（只在有效区域内计算）

---

## ⚙️ GPU配置

当前脚本配置：`GPU="2"`

修改GPU编号：
编辑`run_cylinder_experiments.sh`第60行：
```bash
GPU="2"  # 修改为0, 1, 或2
```

---

## 🐛 故障排除

### 问题1：GPU内存不足

**症状**: `CUDA out of memory`

**解决方案**:
- 减小`--batch_size`：从8改为4
- 减小`--width`：从32改为16
- 减小`--mhc_expansion_ratio`：从4改为2

### 问题2：训练不稳定

**症状**: loss不收敛或NaN

**解决方案**:
- 降低学习率：`--lr 5e-4`
- 启用梯度裁剪：`--max_grad_norm 1.0`
- 增加weight decay：`--weight_decay 1e-3`

### 问题3：FNO不支持use_mhc

**症状**: `TypeError: __init__() got an unexpected keyword argument 'use_mhc'`

**原因**: 使用了错误的conda环境（FNO1）

**解决方案**:
```bash
# 检查当前环境
conda info --envs
echo $CONDA_DEFAULT_ENV

# 应该显示: FNO2

# 切换到FNO2
conda deactivate
conda activate FNO2
```

---

## 📝 数据集说明

- **文件**: Cylinder1.mat
- **路径**: `/home/leeshu/wmm/neuraloperator-main/neuralop/data/datasets/data/Cylinder/Cylinder1.mat`
- **形状**: [747, 20, 3, 64, 64]
  - 样本数：747
  - 时间步数：20
  - 通道数：3（field/vx/mask）
  - 空间分辨率：64×64
- **划分**: 训练集600，测试集147
- **输入**: 10个时间步
- **输出**: 10个时间步

---

## 🎯 实验设计说明

### 训练方式
- **训练时**: Teacher forcing（使用ground truth）
- **评估时**: Autoregressive（使用模型输出）

### 损失函数
- **Per-sample Relative L2**: `||pred - true|| / ||true||`
- **优化器**: Adam
- **学习率调度**: OneCycleLR（10% warmup + cosine annealing）

### 数据增强
- **滑窗数据集**: 从20步序列中提取所有可能的10→1映射
- **每个样本产生10个训练样本**
- **Mask处理**: 输入时应用mask，输出时自然学习边界

---

## 📞 快速参考

### 常用命令

```bash
# 激活环境
conda activate FNO2

# 进入脚本目录
cd /home/leeshu/wmm/neuraloperator-mHC/scripts

# 快速测试
bash train_cylinder_quick_test.sh

# 完整实验
bash run_cylinder_experiments.sh

# 监控进度
bash monitor_experiments.sh

# 评估模型
bash eval_cylinder_mhc.sh

# 单个实验
python train_cylinder_mhc.py --gpu 2 --save_name "test" --epochs 10
```

### 参数快速设置

| 目标 | 参数设置 |
|------|---------|
| 快速验证 | `--epochs 10 --batch_size 16` |
| 标准训练 | `--epochs 500 --batch_size 8` |
| 高性能 | `--epochs 1000 --batch_size 4 --width 64` |
| 禁用mHC | `--use_mhc 0` |
| 推荐mHC | `--use_mhc 1 --mhc_mode "continuous" --mhc_expansion_ratio 4 --mhc_kdb_bandwidth 0.5` |

---

## 🎓 理论背景

### KBS-mHC-FNO架构

KBS-mHC-FNO在标准FNO基础上添加了第三个并行分支：

1. **SpectralConv**: 频域全局积分
2. **ChannelMLP**: 空域局部线性
3. **mHC**: 流形超连接（Manifold Hyper-Connection）

**mHC分支**实现了双随机投影约束，通过Dual-Mode Sinkhorn算法：
- **Discrete模式**: 传统通道级Sinkhorn迭代
- **Continuous模式**: 核密度平衡（KDB）+ 空间平滑

**优势**：
- 增强特征混合能力
- 保持双随机约束的流形结构
- 提升模型表达能力

---

**最后更新**: 2026-03-29
**FNO2环境状态**: ✓ 已配置并验证