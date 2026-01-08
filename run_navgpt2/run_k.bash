#!/bin/bash

# 创建日志目录
mkdir -p logs/navgpt2/feature

echo "========================================="
echo "开始执行推理脚本"
echo "开始时间: $(date)"
echo "========================================="


echo ""
echo "===== 开始串行运行推理脚本 ====="
echo ""

# 脚本1: test_mask_ensemble_inference_2
echo "开始运行脚本1: "
echo "开始时间: $(date)"
nohup sh ./run_navgpt2/test_mask_ensemble_inference_2.bash > logs/navgpt2/feature/test_mask_ensemble_inference_2_2026_1_4_25del.log 2>&1 &
PID1=$!
echo "脚本1 PID: $PID1"

echo "等待脚本1完成..."
wait $PID1
EXIT_CODE1=$?
echo "脚本1完成，退出码: $EXIT_CODE1"
echo "脚本1完成时间: $(date)"

if [ $EXIT_CODE1 -ne 0 ]; then
    echo "警告: 脚本1执行失败，退出码: $EXIT_CODE1"
fi
echo ""

# 脚本2: test_mask_ensemble_inference_3
echo "开始运行脚本2: "
echo "开始时间: $(date)"
nohup sh ./run_navgpt2/test_mask_ensemble_inference_3.bash > logs/navgpt2/feature/test_mask_ensemble_inference_3_2026_1_4_25del.log 2>&1 &
PID2=$!
echo "脚本2 PID: $PID2"

echo "等待脚本2完成..."
wait $PID2
EXIT_CODE2=$?
echo "脚本2完成，退出码: $EXIT_CODE2"
echo "脚本2完成时间: $(date)"

if [ $EXIT_CODE2 -ne 0 ]; then
    echo "警告: 脚本2执行失败，退出码: $EXIT_CODE2"
fi
echo ""

# 脚本3: test_mask_ensemble_inference_4
echo "开始运行脚本3: "
echo "开始时间: $(date)"
nohup sh ./run_navgpt2/test_mask_ensemble_inference_4.bash > logs/navgpt2/feature/test_mask_ensemble_inference_4_2026_1_4_25del.log 2>&1 &
PID3=$!
echo "脚本3 PID: $PID3"

echo ""
echo "===== 所有任务执行完毕 ====="
echo ""


# ==================== 总结 ====================

echo "========================================="
echo "所有推理脚本执行完成"
echo "结束时间: $(date)"
echo "========================================="
echo ""
echo "执行结果总结:"
echo "  脚本1 (test_mask_ensemble_inference_2): 退出码 $EXIT_CODE1"
echo "  脚本2 (test_mask_ensemble_inference_3): 退出码 $EXIT_CODE2"
echo "  脚本3 (test_mask_ensemble_inference_4): 退出码 $EXIT_CODE3"
echo ""

FAILED=0
if [ $EXIT_CODE1 -ne 0 ] || [ $EXIT_CODE2 -ne 0 ] || [ $EXIT_CODE3 -ne 0 ]; then
    FAILED=1
    echo "警告: 有脚本执行失败，请检查日志文件"
fi

exit $FAILED

