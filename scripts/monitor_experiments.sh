#!/bin/bash
# monitor_experiments.sh
# 监控Cylinder实验的运行状态和进度

set -e

# =============================================================================
# 配置
# =============================================================================
LOG_DIR="../logs/Cylinder_mHC"
CKPT_DIR="../checkpoints/Cylinder_mHC"
RESULTS_DIR="../results/Cylinder_mHC"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# 实验列表（11个）
# =============================================================================
declare -a experiments=(
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

# =============================================================================
# 辅助函数
# =============================================================================
get_experiment_status() {
    local exp_name=$1

    # 检查日志文件
    local log_file="$LOG_DIR/${exp_name}.log"

    if [ ! -f "$log_file" ]; then
        echo "NOT_STARTED"
        return
    fi

    # 检查进程是否还在运行
    if grep -q "Total time" "$log_file" 2>/dev/null; then
        echo "COMPLETED"
        return
    fi

    # 检查是否有错误
    if grep -qi "error\|exception\|failed\|traceback" "$log_file" 2>/dev/null | tail -1; then
        echo "FAILED"
        return
    fi

    # 检查最后一个epoch
    local last_epoch=$(grep -o "Epoch *[0-9]*" "$log_file" 2>/dev/null | tail -1 | grep -o "[0-9]*" | tail -1)

    if [ -n "$last_epoch" ]; then
        echo "RUNNING|$last_epoch"
    else
        echo "RUNNING|INIT"
    fi
}

get_experiment_loss() {
    local exp_name=$1
    local log_file="$LOG_DIR/${exp_name}.log"

    if [ ! -f "$log_file" ]; then
        echo "N/A"
        return
    fi

    # 获取最后的test_relL2
    local last_loss=$(grep "test_relL2" "$log_file" 2>/dev/null | tail -1 | grep -o "[0-9]\.[0-9]*e[+-][0-9]*" | tail -1)

    if [ -n "$last_loss" ]; then
        echo "$last_loss"
    else
        echo "N/A"
    fi
}

get_checkpoint_size() {
    local exp_name=$1
    local ckpt_file="$CKPT_DIR/${exp_name}.pt"

    if [ -f "$ckpt_file" ]; then
        local size=$(du -h "$ckpt_file" | cut -f1)
        echo "$size"
    else
        echo "N/A"
    fi
}

get_log_file_size() {
    local exp_name=$1
    local log_file="$LOG_DIR/${exp_name}.log"

    if [ -f "$log_file" ]; then
        local size=$(du -h "$log_file" | cut -f1)
        echo "$size"
    else
        echo "0"
    fi
}

# =============================================================================
# 主监控循环
# =============================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}KBS-mHC-FNO 实验监控${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "监控间隔: 5秒"
echo "日志目录: $LOG_DIR"
echo "按 Ctrl+C 退出监控"
echo ""

# 记录上次状态，用于高亮变化
declare -A last_status
declare -A last_loss

while true; do
    # 清屏
    clear

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}KBS-mHC-FNO 实验监控${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${CYAN}更新时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo ""

    # 表头
    printf "%-25s %12s %12s %12s %10s\n" "实验名称" "状态" "当前Epoch" "Test L2 Loss" "日志大小"
    echo "--------------------------------------------------------------------------------"

    # 统计
    local completed=0
    local running=0
    local failed=0
    local not_started=0

    # 遍历所有实验
    for exp_name in "${experiments[@]}"; do
        # 获取状态
        local status_info=$(get_experiment_status "$exp_name")
        local status=$(echo "$status_info" | cut -d'|' -f1)
        local current_epoch=$(echo "$status_info" | cut -d'|' -f2)

        # 获取loss
        local loss=$(get_experiment_loss "$exp_name")

        # 获取日志大小
        local log_size=$(get_log_file_size "$exp_name")

        # 格式化状态
        local status_color=$NC
        local status_display=$status

        case $status in
            "COMPLETED")
                status_color=$GREEN
                completed=$((completed + 1))
                ;;
            "RUNNING")
                status_color=$YELLOW
                running=$((running + 1))
                ;;
            "FAILED")
                status_color=$RED
                failed=$((failed + 1))
                ;;
            "NOT_STARTED")
                status_color=$BLUE
                not_started=$((not_started + 1))
                ;;
        esac

        # 检查状态是否变化
        if [ "${last_status[$exp_name]}" != "$status" ]; then
            status_display="${status_display}*"
            last_status[$exp_name]=$status
        else
            status_display=$status_display
        fi

        # 检查loss是否变化
        if [ "${last_loss[$exp_name]}" != "$loss" ]; then
            loss_display="${loss}*"
            last_loss[$exp_name]=$loss
        else
            loss_display=$loss
        fi

        # 显示行
        printf "%-25s ${status_color}%-12s${NC} %12s %12s %10s\n" \
            "$exp_name" \
            "$status_display" \
            "${current_epoch:-N/A}" \
            "$loss_display" \
            "$log_size"
    done

    # 统计摘要
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo -e "统计:"
    echo -e "  ${GREEN}已完成: $completed${NC}"
    echo -e "  ${YELLOW}运行中: $running${NC}"
    echo -e "  ${RED}已失败: $failed${NC}"
    echo -e "  ${BLUE}未开始: $not_started${NC}"
    echo ""
    echo -e "${CYAN}总计: $completed / 11 个实验完成${NC}"
    echo ""
    echo -e "${YELLOW}* 表示状态或loss有更新${NC}"

    # GPU状态
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo -e "${CYAN}GPU 状态:${NC}"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | while IFS=, read -r idx name util mem_used mem_total; do
            util_val=${util%.*}
            if (( util_val > 80 )); then
                util_color=$RED
            elif (( util_val > 50 )); then
                util_color=$YELLOW
            else
                util_color=$GREEN
            fi
            printf "  GPU $idx ($name): ${util_color}%3d%% GPU 使用率${NC}, ${util_color}%6s / %6s 显存${NC}\n" "$util" "${mem_used}MiB" "${mem_total}MiB"
        done
    fi

    echo ""
    echo -e "${BLUE}5秒后刷新... (Ctrl+C 退出)${NC}"
    echo ""

    # 等待5秒
    sleep 5
done
