#对比实验
nohup python main.py --model_name SRCNN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_SRCNN_1_L1 2>&1 &


nohup python main.py --model_name ESPCN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_ESPCN_1_L1 2>&1 &


nohup python main.py --model_name VDSR --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_VDSR_1_L1 2>&1 &


nohup python main.py --model_name EDSR --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:1 >log/x4_EDSR_1_L1 2>&1 &


nohup python main.py --model_name MSRN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_1_L1 2>&1 &


nohup python main.py --model_name RCAN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_RCAN_1_L1 2>&1 &


nohup python main.py --model_name RFDN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_RFDN_1_L1 2>&1 &


nohup python main.py --model_name IMDN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_1_L1 2>&1 &


#多少个阶段的对比实验
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG False --patch_size 384 --part 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_1_L1 2>&1 &
#98850
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 --part 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_1_1_L1 2>&1 &
#98880
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 1 --part 4 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_1_1_1_L1 2>&1 &
#98912
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 1 1 --part 2 4 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_1_1_1_1_L1 2>&1 &
#98942
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 1 1 1 --part 2 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_1_1_1_1_1_L1 2>&1 &
#98972
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 1 1 1 1 --part 2 3 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_1_1_1_1_1_1_L1 2>&1 &
#99002
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 1 1 1 1 1 --part 2 3 4 5 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_1_1_1_1_1_1_1_L1 2>&1 &
#99033
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 1 1 1 1 1 1 --part 1 2 3 4 5 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_1_1_1_1_1_1_1_1_L1 2>&1 &
#99063
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 1 1 1 1 1 1 1 --part 1 2 3 4 5 6 7 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_1_1_1_1_1_1_1_1_1_L1 2>&1 &
#99093

#假设统一使用5阶段，重跑划分大小实验
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 12 6 3 1 --part 2 3 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_16_12_6_3_1_1_L1 2>&1 &
#21372
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 8 6 3 1 --part 2 3 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_12_8_6_3_1_1_L1 2>&1 &
#21436
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 8 6 4 3 1 --part 2 3 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_8_6_4_3_1_1_L1 2>&1 &
#21497
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 4 6 8 --part 2 3 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_1_3_4_6_8_1_L1 2>&1 &
#21558
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 8 12 --part 2 3 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_1_3_6_8_12_1_L1 2>&1 &
#21619
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 12 16 --part 2 3 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_1_3_6_12_16_1_L1 2>&1 &
#21680

#在遥感图像上，统一用4阶段，重跑划分大小实验
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 8 3 1 --part 2 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_16_8_3_1_1_L1 2>&1 &
#159492
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --part 2 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_12_6_3_1_1_L1 2>&1 &
#159522
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 8 6 3 1 --part 2 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_8_6_3_1_1_L1 2>&1 &
#159552
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 8 --part 2 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_1_3_6_8_1_L1 2>&1 &
#159582
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 12 --part 2 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_1_3_6_12_1_L1 2>&1 &
#159612
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 8 16 --part 2 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_1_3_8_16_1_L1 2>&1 &
#159642







#验证stride影响
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 12 6 3 1 --part 2 3 4 6 8 --stride 0.5 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_16_12_6_3_1_0p5_L1 2>&1 &
#3333
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 12 6 3 1 --part 2 3 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_16_12_6_3_1_1_L1 2>&1 &
#3383
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 12 6 3 1 --part 2 3 4 6 8 --stride 1.5 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_16_12_6_3_1_1p5_L1 2>&1 &
#3453
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 12 6 3 1 --part 2 3 4 6 8 --stride 2 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_16_12_6_3_1_2_L1 2>&1 &
#3556
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 12 6 3 1 --part 2 3 4 6 8 --stride 2.5 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_16_12_6_3_1_2p5_L1 2>&1 &
#3633



nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 8 3 1 --part 2 4 6 8 --stride 0.5 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_16_8_3_1_0p5_L1 2>&1 &
#198088
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 8 3 1 --part 2 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_16_8_3_1_1_L1 2>&1 &
#198118
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 8 3 1 --part 2 4 6 8 --stride 1.5 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_16_8_3_1_1p5_L1 2>&1 &
#198148
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 8 3 1 --part 2 4 6 8 --stride 2 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_16_8_3_1_2_L1 2>&1 &
#198178
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 8 3 1 --part 2 4 6 8 --stride 2.5 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_16_8_3_1_2p5_L1 2>&1 &
#198208





#最后统一对照实验（按照16,12,6,3,1的模式）
nohup python main.py --model_name SRCNN --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 6 1 --part 1 2 3 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRCNN_PMG_16_6_1_1_L1 2>&1 &
#3736
nohup python main.py --model_name ESPCN --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 6 1 --part 1 2 3 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_ESPCN_PMG_16_6_1_1_L1 2>&1 &
#3787
nohup python main.py --model_name VDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 12 6 3 1 --part 2 3 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_VDSR_PMG_16_12_6_3_1_1_L1 2>&1 &
#3864
nohup python main.py --model_name EDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 12 6 3 1 --part 2 3 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_EDSR_PMG_16_12_6_3_1_1_L1 2>&1 &
#3941
nohup python main.py --model_name MSRN --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 12 6 3 1 --part 2 3 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_MSRN_PMG_16_12_6_3_1_1_L1 2>&1 &

nohup python main.py --model_name MSRN --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 8 3 1 --part 2 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_MSRN_PMG_16_8_3_1_1_L1 2>&1 &
#4016
nohup python main.py --model_name RFDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 6 4 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_RFDN_PMG_6_4_2_1_1_L1 2>&1 &
#4069
nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 12 6 3 1 --part 2 3 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_PMG_16_12_6_3_1_1_L1 2>&1 &

nohup python main.py --model_name SRCNN --scale 4 --is_PMG False --patch_size 384 --part 3 --loss_name 1_L1 --device cuda:0 >log/x4_SRCNN_1_L1 2>&1 &

nohup python main.py --model_name ESPCN --scale 4 --is_PMG False --patch_size 384 --part 3 --loss_name 1_L1 --device cuda:0 >log/x4_ESPCN_1_L1 2>&1 &

nohup python main.py --model_name VDSR --scale 4 --is_PMG False --patch_size 384 --part 8 --loss_name 1_L1 --device cuda:0 >log/x4_VDSR_1_L1 2>&1 &

nohup python main.py --model_name EDSR --scale 4 --is_PMG False --patch_size 384 --part 8 --loss_name 1_L1 --device cuda:0 >log/x4_EDSR_1_L1 2>&1 &

nohup python main.py --model_name MSRN --scale 4 --is_PMG False --patch_size 384 --part 8 --loss_name 1_L1 --device cuda:0 >log/x4_MSRN_1_L1 2>&1 &

nohup python main.py --model_name RFDN --scale 4 --is_PMG False --patch_size 384 --part 4 --loss_name 1_L1 --device cuda:0 >log/x4_RFDN_1_L1 2>&1 &

nohup python main.py --model_name IMDN_plus --scale 4 --is_PMG False --patch_size 384 --part 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG True --patch_size 384 --crop_piece 6 4 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_6_4_2_1_1_L1 2>&1 &

nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_1_L1 2>&1 &

nohup python main.py --model_name Swin2SR --scale 4 --is_PMG True --patch_size 384 --crop_piece 6 4 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_6_4_2_1_1_L1 2>&1 &

nohup python main.py --model_name HAT --scale 4 --is_PMG False --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_1_L1 2>&1 &

nohup python main.py --model_name HAT --scale 4 --is_PMG True --patch_size 384 --crop_piece 6 4 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_6_4_2_1_1_L1 2>&1 &


nohup python main.py --model_name ESRT --scale 4 --is_PMG False --patch_size 384 --part 1 1 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_ESRT_1_L1 2>&1 &

nohup python main.py --model_name ESRT --scale 4 --is_PMG True --patch_size 384 --crop_piece 8 1 --part 1 1 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_ESRT_PMG_8_1_1_L1 2>&1 &

nohup python main.py --model_name BCRN --scale 4 --is_PMG False --patch_size 384 --part 6 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_BCRN_1_L1 2>&1 &

nohup python main.py --model_name BCRN_origin --scale 4 --is_PMG False --patch_size 384 --part 6 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_BCRN_origin_1_L1 2>&1 &

nohup python main.py --model_name BCRN --scale 4 --is_PMG True --patch_size 384 --crop_piece 6 3 1 --part 2 4 6 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_BCRN_PMG_8_1_1_L1 2>&1 &

#测试小图输入是否比大图PMG更优，以SwinIR为基准模型，测试从64到384大小的patch

nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --patch_size 64 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_64_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_96_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --patch_size 128 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_128_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --patch_size 192 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_192_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --patch_size 256 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_256_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_384_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG True --patch_size 384 --crop_piece 6 2 3 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_4_3_2_1_1_L1 2>&1 &










#遥感图像，按照16,8,3,1的模式
nohup python main.py --model_name SRCNN --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 8 1 --part 1 2 3 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRCNN_PMG_16_8_1_1_L1 2>&1 &
#227697
nohup python main.py --model_name ESPCN --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 8 1 --part 1 2 3 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_ESPCN_PMG_16_8_1_1_L1 2>&1 &
#227727
nohup python main.py --model_name VDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 8 3 1 --part 2 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_VDSR_PMG_16_8_3_1_1_L1 2>&1 &
#227633
nohup python main.py --model_name EDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 8 3 1 --part 2 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_EDSR_PMG_16_8_3_1_1_L1 2>&1 &
#227664
nohup python main.py --model_name MSRN --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 8 3 1 --part 2 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_MSRN_PMG_16_8_3_1_1_L1 2>&1 &
#227757
nohup python main.py --model_name RFDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 8 3 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_RFDN_PMG_16_8_3_1_1_L1 2>&1 &
#227787



#APMG实验
nohup python main.py --model_name IMDN_plus_APMG --scale 4 --is_PMG False --patch_size 384 --part 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_APMG_1_L1 2>&1 &
#255648
nohup python main.py --model_name IMDN_plus_APMG --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 8 6 3 1 --part 1 2 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_APMG_12_6_3_1_1_L1 2>&1 &
#255678
nohup python main.py --model_name IMDN_plus_APMG --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 1 1 1 1 --part 1 2 4 6 8 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_plus_APMG_1_1_1_1_1_L1 2>&1 &
#255709

nohup python main.py --model_name ESRT --scale 4 --is_PMG False --patch_size 384 --part 2 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_ESRT_1_L1 2>&1 &

nohup python main.py --model_name ESRT --scale 4 --is_PMG True --patch_size 384 --crop_piece 8 1 --part 1 2 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_ESRT_PMG_8_1_1_L1 2>&1 &



#自适应的位置编码实验
nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape False --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_384_1_L1 2>&1 &

nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape True --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_384_ape_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape False --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_384_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape True --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_384_ape_1_L1 2>&1 &

nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape False --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_384_1_L1 2>&1 &

nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape True --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_384_ape_1_L1 2>&1 &



nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_96_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape True --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_96_ape_1_L1 2>&1 &

nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_96_1_L1 2>&1 &

nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape True --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_96_ape_1_L1 2>&1 &

nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_96_1_L1 2>&1 &

nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape True --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_96_ape_1_L1 2>&1 &



#在更高的patch-size上测试改进的位置编码的效果
nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_96_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape True --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_96_ape_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape False --patch_size 128 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_128_1_L1 2>&1 &
#2278583
nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape True --patch_size 128 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_128_ape_1_L1 2>&1 &
#2278691
nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape False --patch_size 192 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_192_1_L1 2>&1 &
#2278801
nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape True --patch_size 192 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_192_ape_1_L1 2>&1 &
#2278866
nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape False --patch_size 256 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_256_1_L1 2>&1 &
#2279067
nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape True --patch_size 256 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_256_ape_1_L1 2>&1 &
#3495451
nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape False --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_384_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape True --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_384_ape_1_L1 2>&1 &


nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_96_1_L1 2>&1 &
#24713
nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape True --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_96_ape_1_L1 2>&1 &
#24819
nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape False --patch_size 128 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_128_1_L1 2>&1 &
#24918
nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape True --patch_size 128 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_128_ape_1_L1 2>&1 &
#24987
nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape False --patch_size 192 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_192_1_L1 2>&1 &
#25163
nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape True --patch_size 192 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_192_ape_1_L1 2>&1 &
#25231
nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape False --patch_size 256 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_256_1_L1 2>&1 &
#
nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape True --patch_size 256 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_256_ape_1_L1 2>&1 &
#
nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape False --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_384_1_L1 2>&1 &

nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape True --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_384_ape_1_L1 2>&1 &


nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_96_1_L1 2>&1 &
#3495841
nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape True --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_96_ape_1_L1 2>&1 &
#3495938
nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape False --patch_size 128 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_128_1_L1 2>&1 &
#3496298
nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape True --patch_size 128 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_128_ape_1_L1 2>&1 &
#3496470
nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape False --patch_size 192 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_192_1_L1 2>&1 &
#3832256
nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape True --patch_size 192 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_192_ape_1_L1 2>&1 &
#3832361
nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape False --patch_size 256 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_256_1_L1 2>&1 &
#3843964
nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape True --patch_size 256 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_256_ape_1_L1 2>&1 &
#3844099
nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape False --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_384_1_L1 2>&1 &
#
nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape True --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_384_ape_1_L1 2>&1 &
#859

nohup python main.py --model_name HAT --scale 4 --is_PMG True --ape True --patch_size 384 --crop_piece 6 4 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_ape_6_4_2_1_1_L1 2>&1 &

nohup python main.py --model_name HAT --scale 4 --is_PMG True --ape True --patch_size 384 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_ape_6_3_2_1_1_L1 2>&1 &



nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_96_1_L1 2>&1 &

nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape True --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_96_ape_1_L1 2>&1 &

nohup python main.py --model_name HAT --scale 4 --is_PMG True --ape False --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_96_ape_6_3_2_1_1_L1 2>&1 &

nohup python main.py --model_name HAT --scale 4 --is_PMG True --ape True --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_96_ape_6_3_2_1_1_L1 2>&1 &


nohup python main.py --model_name SRFormer --scale 4 --is_PMG False --ape False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRFormer_96_1_L1 2>&1 &
#2830766
nohup python main.py --model_name SRFormer --scale 4 --is_PMG False --ape True --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRFormer_96_ape_1_L1 2>&1 &
#2830870
nohup python main.py --model_name SRFormer --scale 4 --is_PMG False --ape False --patch_size 128 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRFormer_128_1_L1 2>&1 &
#2831086
nohup python main.py --model_name SRFormer --scale 4 --is_PMG False --ape True --patch_size 128 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRFormer_128_ape_1_L1 2>&1 &
#2831192
nohup python main.py --model_name SRFormer --scale 4 --is_PMG False --ape False --patch_size 192 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRFormer_192_1_L1 2>&1 &
#2831291
nohup python main.py --model_name SRFormer --scale 4 --is_PMG False --ape True --patch_size 192 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRFormer_192_ape_1_L1 2>&1 &
#2831390
nohup python main.py --model_name SRFormer --scale 4 --is_PMG False --ape False --patch_size 256 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRFormer_256_1_L1 2>&1 &
#2831528
nohup python main.py --model_name SRFormer --scale 4 --is_PMG False --ape True --patch_size 256 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRFormer_256_ape_1_L1 2>&1 &
#
nohup python main.py --model_name SRFormer --scale 4 --is_PMG False --ape False --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRFormer_384_1_L1 2>&1 &
#
nohup python main.py --model_name SRFormer --scale 4 --is_PMG False --ape True --patch_size 384 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRFormer_384_ape_1_L1 2>&1 &



nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_96_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG False --ape True --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_96_ape_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG True --ape False --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_96_6_3_2_1_1_L1 2>&1 &

nohup python main.py --model_name SwinIR --scale 4 --is_PMG True --ape True --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SwinIR_96_ape_6_3_2_1_1_L1 2>&1 &



nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_96_1_L1 2>&1 &
#24713
nohup python main.py --model_name Swin2SR --scale 4 --is_PMG False --ape True --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_96_ape_1_L1 2>&1 &

nohup python main.py --model_name Swin2SR --scale 4 --is_PMG True --ape False --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_96_6_3_2_1_1_L1 2>&1 &

nohup python main.py --model_name Swin2SR --scale 4 --is_PMG True --ape True --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_Swin2SR_96_ape_6_3_2_1_1_L1 2>&1 &



nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_96_1_L1 2>&1 &

nohup python main.py --model_name HAT --scale 4 --is_PMG False --ape True --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_96_ape_1_L1 2>&1 &

nohup python main.py --model_name HAT --scale 4 --is_PMG True --ape False --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_96_6_3_2_1_1_L1 2>&1 &

nohup python main.py --model_name HAT --scale 4 --is_PMG True --ape True --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HAT_96_ape_6_3_2_1_1_L1 2>&1 &



nohup python main.py --model_name SRFormer --scale 4 --is_PMG False --ape False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRFormer_96_1_L1 2>&1 &

nohup python main.py --model_name SRFormer --scale 4 --is_PMG False --ape True --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRFormer_96_ape_1_L1 2>&1 &

nohup python main.py --model_name SRFormer --scale 4 --is_PMG True --ape False --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRFormer_96_6_3_2_1_1_L1 2>&1 &

nohup python main.py --model_name SRFormer --scale 4 --is_PMG True --ape True --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRFormer_96_ape_6_3_2_1_1_L1 2>&1 &



nohup python main.py --model_name HiT_SIR --scale 4 --is_PMG False --ape False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HiT_SIR_96_1_L1 2>&1 &
#2830766
nohup python main.py --model_name HiT_SIR --scale 4 --is_PMG False --ape True --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HiT_SIR_96_ape_1_L1 2>&1 &

nohup python main.py --model_name HiT_SIR --scale 4 --is_PMG True --ape False --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HiT_SIR_96_6_3_2_1_1_L1 2>&1 &

nohup python main.py --model_name HiT_SIR --scale 4 --is_PMG True --ape True --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HiT_SIR_96_ape_6_3_2_1_1_L1 2>&1 &



nohup python main.py --model_name HiT_SNG --scale 4 --is_PMG False --ape False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HiT_SNG_96_1_L1 2>&1 &
#2830766
nohup python main.py --model_name HiT_SNG --scale 4 --is_PMG False --ape True --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HiT_SNG_96_ape_1_L1 2>&1 &

nohup python main.py --model_name HiT_SNG --scale 4 --is_PMG True --ape False --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HiT_SNG_96_6_3_2_1_1_L1 2>&1 &

nohup python main.py --model_name HiT_SNG --scale 4 --is_PMG True --ape True --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HiT_SNG_96_ape_6_3_2_1_1_L1 2>&1 &



nohup python main.py --model_name HiT_SRF --scale 4 --is_PMG False --ape False --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HiT_SRF_96_1_L1 2>&1 &
#2830766
nohup python main.py --model_name HiT_SRF --scale 4 --is_PMG False --ape True --patch_size 96 --part 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HiT_HiT_SRF_96_ape_1_L1 2>&1 &

nohup python main.py --model_name HiT_SRF --scale 4 --is_PMG True --ape False --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HiT_SRF_96_6_3_2_1_1_L1 2>&1 &

nohup python main.py --model_name HiT_SRF --scale 4 --is_PMG True --ape True --patch_size 96 --crop_piece 6 3 2 1 --part 1 2 3 4 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_HiT_SRF_96_ape_6_3_2_1_1_L1 2>&1 &