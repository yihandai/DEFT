#!/bin/bash

# 创建日志目录
mkdir -p logs/navgpt/feature

# 记录开始时间
echo "========================================="
echo "开始执行实验脚本"
echo "开始时间: $(date)"
echo "========================================="

# ==================== 串行运行脚本，顺序: 1, 3, 2, 4, 5, 7, 6, 8 ====================

# 脚本1: test_mask_ensemble_update_5
echo ""
echo "===== 开始运行脚本1: test_mask_ensemble_update_4 ====="
echo "开始时间: $(date)"
nohup sh run_navgpt/test_mask_ensemble_update_4.bash > logs/navgpt/feature/update_4.log 2>&1 &
PID1=$!
echo "脚本1 PID: $PID1"
wait $PID1
EXIT_CODE1=$?
echo "脚本1完成，退出码: $EXIT_CODE1"
echo "完成时间: $(date)"
if [ $EXIT_CODE1 -ne 0 ]; then
    echo "警告: 脚本1执行失败，退出码: $EXIT_CODE1"
fi

# 脚本3: test_mask_ensemble_inference_5
echo ""
echo "===== 开始运行脚本3: test_mask_ensemble_inference_4 ====="
echo "开始时间: $(date)"
nohup sh run_navgpt/test_mask_ensemble_inference_4.bash > logs/navgpt/feature/inference_4.log 2>&1 &
PID3=$!
echo "脚本3 PID: $PID3"
wait $PID3
EXIT_CODE3=$?
echo "脚本3完成，退出码: $EXIT_CODE3"
echo "完成时间: $(date)"
if [ $EXIT_CODE3 -ne 0 ]; then
    echo "警告: 脚本3执行失败，退出码: $EXIT_CODE3"
fi

# 脚本2: test_mask_ensemble_update_4
echo ""
echo "===== 开始运行脚本2: test_mask_ensemble_update_5 ====="
echo "开始时间: $(date)"
nohup sh run_navgpt/test_mask_ensemble_update_5.bash > logs/navgpt/feature/update_5.log 2>&1 &
PID2=$!
echo "脚本2 PID: $PID2"
wait $PID2
EXIT_CODE2=$?
echo "脚本2完成，退出码: $EXIT_CODE2"
echo "完成时间: $(date)"
if [ $EXIT_CODE2 -ne 0 ]; then
    echo "警告: 脚本2执行失败，退出码: $EXIT_CODE2"
fi

# 脚本4: test_mask_ensemble_inference_4
echo ""
echo "===== 开始运行脚本4: test_mask_ensemble_inference_5 ====="
echo "开始时间: $(date)"
nohup sh run_navgpt/test_mask_ensemble_inference_5.bash > logs/navgpt/feature/inference_5.log 2>&1 &
PID4=$!
echo "脚本4 PID: $PID4"
wait $PID4
EXIT_CODE4=$?
echo "脚本4完成，退出码: $EXIT_CODE4"
echo "完成时间: $(date)"
if [ $EXIT_CODE4 -ne 0 ]; then
    echo "警告: 脚本4执行失败，退出码: $EXIT_CODE4"
fi

# 脚本5: test_mask_ensemble_update_3
echo ""
echo "===== 开始运行脚本5: test_mask_ensemble_update_3 ====="
echo "开始时间: $(date)"
nohup sh run_navgpt/test_mask_ensemble_update_3.bash > logs/navgpt/feature/update_3.log 2>&1 &
PID5=$!
echo "脚本5 PID: $PID5"
wait $PID5
EXIT_CODE5=$?
echo "脚本5完成，退出码: $EXIT_CODE5"
echo "完成时间: $(date)"
if [ $EXIT_CODE5 -ne 0 ]; then
    echo "警告: 脚本5执行失败，退出码: $EXIT_CODE5"
fi

# 脚本7: test_mask_ensemble_inference_3
echo ""
echo "===== 开始运行脚本7: test_mask_ensemble_inference_3 ====="
echo "开始时间: $(date)"
nohup sh run_navgpt/test_mask_ensemble_inference_3.bash > logs/navgpt/feature/inference_3.log 2>&1 &
PID7=$!
echo "脚本7 PID: $PID7"
wait $PID7
EXIT_CODE7=$?
echo "脚本7完成，退出码: $EXIT_CODE7"
echo "完成时间: $(date)"
if [ $EXIT_CODE7 -ne 0 ]; then
    echo "警告: 脚本7执行失败，退出码: $EXIT_CODE7"
fi

# 脚本6: test_mask_ensemble_update_2
echo ""
echo "===== 开始运行脚本6: test_mask_ensemble_update_2 ====="
echo "开始时间: $(date)"
nohup sh run_navgpt/test_mask_ensemble_update_2.bash > logs/navgpt/feature/update_2.log 2>&1 &
PID6=$!
echo "脚本6 PID: $PID6"
wait $PID6
EXIT_CODE6=$?
echo "脚本6完成，退出码: $EXIT_CODE6"
echo "完成时间: $(date)"
if [ $EXIT_CODE6 -ne 0 ]; then
    echo "警告: 脚本6执行失败，退出码: $EXIT_CODE6"
fi

# 脚本8: test_mask_ensemble_inference_2
echo ""
echo "===== 开始运行脚本8: test_mask_ensemble_inference_2 ====="
echo "开始时间: $(date)"
nohup sh run_navgpt/test_mask_ensemble_inference_2.bash > logs/navgpt/feature/inference_2.log 2>&1 &
PID8=$!
echo "脚本8 PID: $PID8"
wait $PID8
EXIT_CODE8=$?
echo "脚本8完成，退出码: $EXIT_CODE8"
echo "完成时间: $(date)"
if [ $EXIT_CODE8 -ne 0 ]; then
    echo "警告: 脚本8执行失败，退出码: $EXIT_CODE8"
fi

# ==================== 总结 ====================

echo "========================================="
echo "所有实验脚本执行完成"
echo "结束时间: $(date)"
echo "========================================="
echo ""
echo "执行结果总结:"
echo "  脚本1 (test_mask_ensemble_update_5):        退出码 $EXIT_CODE1"
echo "  脚本2 (test_mask_ensemble_update_4):       退出码 $EXIT_CODE2"
echo "  脚本3 (test_mask_ensemble_inference_5):     退出码 $EXIT_CODE3"
echo "  脚本4 (test_mask_ensemble_inference_4):     退出码 $EXIT_CODE4"
echo "  脚本5 (test_mask_ensemble_update_3):        退出码 $EXIT_CODE5"
echo "  脚本6 (test_mask_ensemble_update_2):        退出码 $EXIT_CODE6"
echo "  脚本7 (test_mask_ensemble_inference_3):     退出码 $EXIT_CODE7"
echo "  脚本8 (test_mask_ensemble_inference_2):     退出码 $EXIT_CODE8"
echo ""

# 检查是否有失败的脚本
FAILED=0
if [ $EXIT_CODE1 -ne 0 ] || [ $EXIT_CODE2 -ne 0 ] || [ $EXIT_CODE3 -ne 0 ] || \
   [ $EXIT_CODE4 -ne 0 ] || [ $EXIT_CODE5 -ne 0 ] || [ $EXIT_CODE6 -ne 0 ] || \
   [ $EXIT_CODE7 -ne 0 ] || [ $EXIT_CODE8 -ne 0 ]; then
    FAILED=1
    echo "警告: 有脚本执行失败，请检查日志文件"
fi

exit $FAILED

