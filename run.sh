
#对比实验
nohup python main.py --model_name SRCNN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_SRCNN_1_L1 2>&1 &
#30948
nohup python main.py --model_name SRCNN --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 6 12 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRCNN_PMG_1_6_12_1_L1 2>&1 &
#31027
nohup python main.py --model_name SRCNN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 1 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRCNN_PMG_12_6_1_1_L1 2>&1 &
#31076
nohup python main.py --model_name SRCNN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 1 --stride 2 --loss_name 1_L1 --device cuda:0 >log/x4_SRCNN_PMG_12_6_1_2_L1 2>&1 &
#
nohup python main.py --model_name SRCNN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 1 --stride 3 --loss_name 1_L1 --device cuda:0 >log/x4_SRCNN_PMG_12_6_1_3_L1 2>&1 &
#

nohup python main.py --model_name ESPCN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_ESPCN_1_L1 2>&1 &
#31182
nohup python main.py --model_name ESPCN --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 6 12 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_ESPCN_PMG_1_6_12_1_L1 2>&1 &
#31259
nohup python main.py --model_name ESPCN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 1 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_ESPCN_PMG_12_6_1_1_L1 2>&1 &
#31310
nohup python main.py --model_name ESPCN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 1 --stride 2 --loss_name 1_L1 --device cuda:0 >log/x4_ESPCN_PMG_12_6_1_2_L1 2>&1 &
#31310
nohup python main.py --model_name ESPCN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 1 --stride 3 --loss_name 1_L1 --device cuda:0 >log/x4_ESPCN_PMG_12_6_1_3_L1 2>&1 &
#31310

nohup python main.py --model_name VDSR --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_VDSR_1_L1 2>&1 &
#31414
nohup python main.py --model_name VDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 6 12 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_VDSR_PMG_1_6_12_1_L1 2>&1 &
#31498
nohup python main.py --model_name VDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 1 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_VDSR_PMG_12_6_1_1_L1 2>&1 &
#31549
nohup python main.py --model_name VDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 1 --stride 2 --loss_name 1_L1 --device cuda:0 >log/x4_VDSR_PMG_12_6_1_2_L1 2>&1 &
#31549
nohup python main.py --model_name VDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 1 --stride 3 --loss_name 1_L1 --device cuda:0 >log/x4_VDSR_PMG_12_6_1_3_L1 2>&1 &
#31549

nohup python main.py --model_name EDSR --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:1 >log/x4_EDSR_1_L1 2>&1 &
#31656
nohup python main.py --model_name EDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 12 --stride 1 --loss_name 1_L1 --device cuda:1 >log/x4_EDSR_PMG_1_3_6_12_1_L1 2>&1 &
#31707
nohup python main.py --model_name EDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 1 --loss_name 1_L1 --device cuda:1 >log/x4_EDSR_PMG_12_6_3_1_1_L1 2>&1 &
#31784
nohup python main.py --model_name EDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 2 --loss_name 1_L1 --device cuda:1 >log/x4_EDSR_PMG_12_6_3_1_2_L1 2>&1 &
#31784
nohup python main.py --model_name EDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 3 --loss_name 1_L1 --device cuda:1 >log/x4_EDSR_PMG_12_6_3_1_3_L1 2>&1 &
#31784

nohup python main.py --model_name MSRN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_1_L1 2>&1 &
#31892
nohup python main.py --model_name MSRN --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 12 --stride 1 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_PMG_1_3_6_12_1_L1 2>&1 &
#31943
nohup python main.py --model_name MSRN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 1 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_PMG_12_6_3_1_1_L1 2>&1 &
#32018
nohup python main.py --model_name MSRN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 2 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_PMG_12_6_3_1_2_L1 2>&1 &
#105535
nohup python main.py --model_name MSRN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 3 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_PMG_12_6_3_1_3_L1 2>&1 &
#105612

nohup python main.py --model_name RCAN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_RCAN_1_L1 2>&1 &
#51888
nohup python main.py --model_name RCAN --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 12 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_RCAN_PMG_1_3_6_12_1_L1 2>&1 &
#19580
nohup python main.py --model_name RCAN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 1 --loss_name 1_L1 --device cuda:1 >log/x4_RCAN_PMG_12_6_3_1_1_L1 2>&1 &
#19661
nohup python main.py --model_name RCAN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 2 --loss_name 1_L1 --device cuda:0 >log/x4_RCAN_PMG_12_6_3_1_2_L1 2>&1 &
#19712
nohup python main.py --model_name RCAN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 3 --loss_name 1_L1 --device cuda:1 >log/x4_RCAN_PMG_12_6_3_1_3_L1 2>&1 &
#19789

nohup python main.py --model_name RFDN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_RFDN_1_L1 2>&1 &
#167917
nohup python main.py --model_name RFDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 6 4 3 1 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_RFDN_PMG_6_4_3_1_1_L1 2>&1 &
#167996
nohup python main.py --model_name RFDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 4 6 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_RFDN_PMG_1_3_4_6_1_L1 2>&1 &
#168075
nohup python main.py --model_name RFDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 6 4 3 1 --stride 2 --loss_name 1_L1 --device cuda:1 >log/x4_RFDN_PMG_6_4_3_1_2_L1 2>&1 &
#168126
nohup python main.py --model_name RFDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 6 4 3 1 --stride 3 --loss_name 1_L1 --device cuda:1 >log/x4_RFDN_PMG_6_4_3_1_3_L1 2>&1 &
#168203


nohup python main.py --model_name IMDN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_1_L1 2>&1 &
#12450
nohup python main.py --model_name IMDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_PMG_12_6_3_1_1_L1 2>&1 &
#12506
nohup python main.py --model_name IMDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 12 --stride 1 --loss_name 1_L1 --device cuda:1 >log/x4_IMDN_PMG_1_3_6_12_1_L1 2>&1 &
#12557
nohup python main.py --model_name IMDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 2 --loss_name 1_L1 --device cuda:1 >log/x4_IMDN_PMG_12_6_3_1_2_L1 2>&1 &
#12684
nohup python main.py --model_name IMDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 3 --loss_name 1_L1 --device cuda:1 >log/x4_IMDN_PMG_12_6_3_1_3_L1 2>&1 &
#12737

#切割大小的对比实验
nohup python main.py --model_name IMDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 12 6 1 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_PMG_16_12_6_1_1_L1 2>&1 &
#
nohup python main.py --model_name IMDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_PMG_12_6_3_1_1_L1 2>&1 &
#
nohup python main.py --model_name IMDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 6 4 3 1 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_IMDN_PMG_6_4_3_1_1_L1 2>&1 &
#
nohup python main.py --model_name IMDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 4 6 --stride 1 --loss_name 1_L1 --device cuda:1 >log/x4_IMDN_PMG_1_3_4_6_1_L1 2>&1 &
#
nohup python main.py --model_name IMDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 12 --stride 1 --loss_name 1_L1 --device cuda:1 >log/x4_IMDN_PMG_1_3_6_12_1_L1 2>&1 &
#
nohup python main.py --model_name IMDN --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 6 12 16 --stride 1 --loss_name 1_L1 --device cuda:1 >log/x4_IMDN_PMG_1_6_12_16_1_L1 2>&1 &
#


#多少个阶段的对比实验