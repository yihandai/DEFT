#!/bin/bash

# 创建日志目录
mkdir -p logs/navgpt/feature

echo "========================================="
echo "开始执行实验脚本"
echo "开始时间: $(date)"
echo "========================================="

echo ""
echo "===== 阶段1: 并行运行脚本 1、2 ====="
echo ""

# 脚本1: train_bagging_0 并行
echo "开始运行脚本1: train_bagging_0"
echo "开始时间: $(date)"
nohup sh ./run_navgpt2/train_bagging_0.bash > logs/navgpt2/bagging/navgpt2_bagging_0_$(date +%Y_%m_%d_%H_%M_%S).log 2>&1 &
PID1=$!
echo "脚本1 PID: $PID1"

# 脚本2: train_bagging_1 并行
echo "开始运行脚本2: train_bagging_1"
echo "开始时间: $(date)"
nohup sh ./run_navgpt2/train_bagging_1.bash > logs/navgpt2/bagging/navgpt2_bagging_1_$(date +%Y_%m_%d_%H_%M_%S).log 2>&1 &
PID2=$!
echo "脚本2 PID: $PID2"

echo "等待脚本1和脚本2完成..."
wait $PID1
EXIT_CODE1=$?
wait $PID2
EXIT_CODE2=$?
echo "脚本1完成，退出码: $EXIT_CODE1"
echo "脚本2完成，退出码: $EXIT_CODE2"
echo "阶段1完成时间: $(date)"

if [ $EXIT_CODE1 -ne 0 ]; then
    echo "警告: 脚本1执行失败，退出码: $EXIT_CODE1"
fi
if [ $EXIT_CODE2 -ne 0 ]; then
    echo "警告: 脚本2执行失败，退出码: $EXIT_CODE2"
fi
echo ""
echo "===== 阶段1完成，已确认1和2都完成 ====="
echo ""

echo "===== 阶段2: 并行运行脚本 3、4 ====="
echo ""

# 脚本3: train_bagging_2 并行
echo "开始运行脚本3: train_bagging_2"
echo "开始时间: $(date)"
nohup sh ./run_navgpt2/train_bagging_2.bash > logs/navgpt2/bagging/navgpt2_bagging_2_$(date +%Y_%m_%d_%H_%M_%S).log 2>&1 &
PID3=$!
echo "脚本3 PID: $PID3"

# 脚本4: train_bagging_3 并行
echo "开始运行脚本4: train_bagging_3"
echo "开始时间: $(date)"
nohup sh ./run_navgpt2/train_bagging_3.bash > logs/navgpt2/bagging/navgpt2_bagging_3_$(date +%Y_%m_%d_%H_%M_%S).log 2>&1 &
PID4=$!
echo "脚本4 PID: $PID4"

echo "等待脚本3和脚本4完成..."
wait $PID3
EXIT_CODE3=$?
wait $PID4
EXIT_CODE4=$?
echo "脚本3完成，退出码: $EXIT_CODE3"
echo "脚本4完成，退出码: $EXIT_CODE4"
echo "阶段2完成时间: $(date)"

if [ $EXIT_CODE3 -ne 0 ]; then
    echo "警告: 脚本3执行失败，退出码: $EXIT_CODE3"
fi
if [ $EXIT_CODE4 -ne 0 ]; then
    echo "警告: 脚本4执行失败，退出码: $EXIT_CODE4"
fi
echo ""
echo "===== 阶段2完成，已确认3和4都完成 ====="
echo ""

# 阶段3: 顺序运行脚本5
echo "===== 阶段3: 运行脚本 5 ====="
echo ""
echo "开始运行脚本5: train_bagging_4"
echo "开始时间: $(date)"
nohup sh ./run_navgpt2/train_bagging_4.bash > logs/navgpt2/bagging/navgpt2_bagging_4_$(date +%Y_%m_%d_%H_%M_%S).log 2>&1 &
PID5=$!
echo "脚本5 PID: $PID5"

wait $PID5
EXIT_CODE5=$?
echo "脚本5完成，退出码: $EXIT_CODE5"
echo "阶段3完成时间: $(date)"
if [ $EXIT_CODE5 -ne 0 ]; then
    echo "警告: 脚本5执行失败，退出码: $EXIT_CODE5"
fi
echo ""

echo ""
echo "===== 所有任务执行完毕 ====="
echo ""

# ==================== 总结 ====================

echo "========================================="
echo "所有实验脚本执行完成"
echo "结束时间: $(date)"
echo "========================================="
echo ""
echo "执行结果总结:"
echo "  脚本1 (train_bagging_0): 退出码 $EXIT_CODE1"
echo "  脚本2 (train_bagging_1): 退出码 $EXIT_CODE2"
echo "  脚本3 (train_bagging_2): 退出码 $EXIT_CODE3"
echo "  脚本4 (train_bagging_3): 退出码 $EXIT_CODE4"
echo "  脚本5 (train_bagging_4): 退出码 $EXIT_CODE5"
echo ""

FAILED=0
if [ $EXIT_CODE1 -ne 0 ] || [ $EXIT_CODE2 -ne 0 ] || [ $EXIT_CODE3 -ne 0 ] || \
   [ $EXIT_CODE4 -ne 0 ] || [ $EXIT_CODE5 -ne 0 ]; then
    FAILED=1
    echo "警告: 有脚本执行失败，请检查日志文件"
fi

exit $FAILED
