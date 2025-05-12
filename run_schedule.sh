#!/bin/bash

# 执行第一个Python脚本，并在后台运行
nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_1_L1 2>&1 &
first_pid=$!

# 等待第一个脚本执行完成
wait $first_pid

# 执行第二个Python脚本，并在后台运行
nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_1_L1 2>&1 &
second_pid=$!

# 等待第二个脚本执行完成
wait $second_pid

nohup python main.py --model_name ESRT --scale 4 --is_PMG False --patch_size 384 --part 2 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_ESRT_1_L1 2>&1 &

third_pid=$!

wait $third_pid

nohup python main.py --model_name HAT --scale 4 --is_PMG False --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_1_L1 2>&1 &

forth_pid=$!

wait $forth_pid