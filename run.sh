python prepare_datasets.py

#nohup python main.py --model_name SRCNN --scale 4 --device cuda:0 >log/x4_SRCNN_1_L1 2>&1 &
##3104404
#nohup python main.py --model_name SRCNN --scale 8 --device cuda:0 >log/x8_SRCNN_1_L1 2>&1 &
##3104456
#nohup python main.py --model_name SRCNN --scale 16 --device cuda:0 >log/x16_SRCNN_1_L1 2>&1 &
##3104533

#nohup python main.py --model_name FSRCNN --scale 4 --device cuda:0 >log/x4_FSRCNN_1_L1 2>&1 &
##3130370
#nohup python main.py --model_name FSRCNN --scale 8 --device cuda:0 >log/x8_FSRCNN_1_L1 2>&1 &
##3130425
nohup python main.py --model_name FSRCNN --scale 16 --device cuda:0 >log/x16_FSRCNN_1_L1 2>&1 &
#3130506

#nohup python main.py --model_name ESPCN --scale 4 --device cuda:0 >log/x4_ESPCN_1_L1 2>&1 &
##3382749
#nohup python main.py --model_name ESPCN --scale 8 --device cuda:0 >log/x8_ESPCN_1_L1 2>&1 &
##3382752
#nohup python main.py --model_name ESPCN --scale 16 --device cuda:0 >log/x16_ESPCN_1_L1 2>&1 &
##3382902

#nohup python main.py --model_name MSRN --scale 4 --device cuda:0 >log/x4_MSRN_1_L1 2>&1 &
##2991019
#nohup python main.py --model_name MSRN --scale 8 --device cuda:0 >log/x8_MSRN_1_L1 2>&1 &
##2991094
#nohup python main.py --model_name MSRN --scale 16 --device cuda:0 >log/x16_MSRN_1_L1 2>&1 &
##2991147

#nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG False --loss_name 1_L1 --device cuda:0 >log/x4_MSRN_PMG_1_L1 2>&1 &
##3453708
#nohup python main.py --model_name MSRN_PMG --scale 8 --is_PMG False --loss_name 1_L1 --device cuda:0 >log/x8_MSRN_PMG_1_L1 2>&1 &
##3453802
#nohup python main.py --model_name MSRN_PMG --scale 16 --is_PMG False --loss_name 1_L1 --device cuda:0 >log/x16_MSRN_PMG_1_L1 2>&1 &
##3453936

nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG True --is_crop False --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_PMG_PMG_no_crop_1_L1 2>&1 &
#3454039
nohup python main.py --model_name MSRN_PMG --scale 8 --is_PMG True --is_crop False --loss_name 1_L1 --device cuda:1 >log/x8_MSRN_PMG_PMG_no_crop_1_L1 2>&1 &
#3454090
nohup python main.py --model_name MSRN_PMG --scale 16 --is_PMG True --is_crop False --loss_name 1_L1 --device cuda:1 >log/x16_MSRN_PMG_PMG_no_crop_1_L1 2>&1 &
#3454167

nohup python PMG_main.py --model_name MSRN_OLD_PMG --scale 4 --is_PMG True --loss_name 1_L1 --device cuda:0 >log/x4_MSRN_OLD_PMG_PMG_1_L1 2>&1 &
#3651810
nohup python PMG_main.py --model_name MSRN_OLD_PMG --scale 8 --is_PMG True --loss_name 1_L1 --device cuda:0 >log/x8_MSRN_OLD_PMG_PMG_1_L1 2>&1 &
#3651885
nohup python PMG_main.py --model_name MSRN_OLD_PMG --scale 16 --is_PMG True --loss_name 1_L1 --device cuda:0 >log/x16_MSRN_OLD_PMG_PMG_1_L1 2>&1 &
#3651941

nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG True --loss_name 1_L1 --device cuda:0 >log/x4_MSRN_PMG_PMG_1_L1 2>&1 &
#127392
nohup python main.py --model_name MSRN_PMG --scale 8 --is_PMG True --loss_name 1_L1 --device cuda:1 >log/x8_MSRN_PMG_PMG_1_L1 2>&1 &
#127570

#python main.py --model_name RCAN --scale 2 --n_resgroups 10 --n_resblocks 20
#python main.py --model_name RCAN --scale 3 --n_resgroups 10 --n_resblocks 20
#python main.py --model_name RCAN --scale 4 --n_resgroups 10 --n_resblocks 20
#python main.py --model_name RCAN --scale 8 --n_resgroups 10 --n_resblocks 20

nohup python main.py --model_name RCAN --scale 4 --loss_name 1_L1 --n_resgroups 10 --n_resblocks 20 --device cuda:0 >log/x4_RCAN_1_L1 2>&1 &

nohup python main.py --model_name RCAN_PMG --scale 4 --is_PMG True --is_crop True --loss_name 1_L1 --n_resgroups 10 --n_resblocks 20 --device cuda:1 >log/x4_RCAN_PMG_PMG_1_L1 2>&1 &

#划分N个stage的实验，N=1,2,3,4，对应crop大小[不使用PMG，[16]，[16,8],[16,8,4]]
nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG False --loss_name 1_L1 --device cuda:0 >log/x4_MSRN_PMG_1_L1_1 2>&1 &
#3532397
nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG True --crop_piece 16 --stride 0.5 --loss_name 1_L1 --device cuda:0 >log/x4_MSRN_PMG_PMG_16_1_L1 2>&1 &
#3532658
nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG True --crop_piece 16 8 --stride 0.5 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_PMG_PMG_16_8_1_L1 2>&1 &
#3532760
nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG True --crop_piece 16 8 4 --stride 0.5 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_PMG_PMG_16_8_4_1_L1 2>&1 &
#3532860

#划分大小的实验（统一使用4个阶段）
nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG True --patch_size 384 --crop_piece 16 8 4 1 --stride 0.5 --loss_name 1_L1 --device cuda:0 >log/x4_MSRN_PMG_PMG_16_8_4_1_L1 2>&1 &
#1049153
nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 4 8 16 --stride 0.5 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_PMG_PMG_1_4_8_16_L1 2>&1 &
#1049236
nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 0.5 --loss_name 1_L1 --device cuda:0 >log/x4_MSRN_PMG_PMG_12_6_3_1_L1 2>&1 &
#1049311
nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 12 --stride 0.5 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_PMG_PMG_1_3_6_12_L1 2>&1 &
#1049364

#补充stride=1时结果是否更优
nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 4 8 16 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_MSRN_PMG_PMG_1_1_4_8_16_L1 2>&1 &
#2215914
nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 12 --stride 1 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_PMG_PMG_1_1_3_6_12_L1 2>&1 &
#2215993
#另外再实验2倍步进和3倍步进是否更优
nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 12 --stride 2 --loss_name 1_L1 --device cuda:0 >log/x4_MSRN_PMG_PMG_2_1_3_6_12_L1 2>&1 &
#2216046
nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 12 --stride 3 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_PMG_PMG_2_1_3_6_12_L1 2>&1 &
#2216123

#实验meta-learning
nohup python meta_main.py --model_name MSRN_PMG --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 4 8 16 --stride 2 --loss_name 1_spl --device cuda:0 >log/x4_meta_MSRN_PMG_PMG_1_1_4_8_16_spl 2>&1 &
#6422
nohup python meta_main.py --model_name MSRN_PMG --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 12 --stride 2 --loss_name 1_spl --device cuda:1 >log/x4_meta_MSRN_PMG_PMG_1_1_3_6_12_spl 2>&1 &
#6500

nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 4 8 16 --stride 2 --loss_name 1_L1 --device cuda:0 >log/x4_MSRN_PMG_PMG_1_1_4_8_16_L1 2>&1 &
#5882
nohup python main.py --model_name MSRN_PMG --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 4 8 16 --stride 3 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_PMG_PMG_1_1_4_8_16_L1 2>&1 &
#5959

#对比实验
nohup python main.py --model_name SRCNN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_SRCNN_1_L1 2>&1 &
#30948
nohup python main.py --model_name SRCNN --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 6 12 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRCNN_PMG_1_6_12_1_L1 2>&1 &
#31027
nohup python main.py --model_name SRCNN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 1 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_SRCNN_PMG_12_6_1_1_L1 2>&1 &
#31076

nohup python main.py --model_name ESPCN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_ESPCN_1_L1 2>&1 &
#31182
nohup python main.py --model_name ESPCN --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 6 12 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_ESPCN_PMG_1_6_12_1_L1 2>&1 &
#31259
nohup python main.py --model_name ESPCN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 1 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_ESPCN_PMG_12_6_1_1_L1 2>&1 &
#31310

nohup python main.py --model_name VDSR --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_VDSR_1_L1 2>&1 &
#31414
nohup python main.py --model_name VDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 6 12 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_VDSR_PMG_1_6_12_1_L1 2>&1 &
#31498
nohup python main.py --model_name VDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 1 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_VDSR_PMG_12_6_1_1_L1 2>&1 &
#31549

nohup python main.py --model_name EDSR --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:1 >log/x4_EDSR_1_L1 2>&1 &
#31656
nohup python main.py --model_name EDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 12 --stride 1 --loss_name 1_L1 --device cuda:1 >log/x4_EDSR_PMG_1_3_6_12_1_L1 2>&1 &
#31707
nohup python main.py --model_name EDSR --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 1 --loss_name 1_L1 --device cuda:1 >log/x4_EDSR_PMG_12_6_3_1_1_L1 2>&1 &
#31784

nohup python main.py --model_name MSRN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_1_L1 2>&1 &
#31892
nohup python main.py --model_name MSRN --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 12 --stride 1 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_PMG_1_3_6_12_1_L1 2>&1 &
#31943
nohup python main.py --model_name MSRN --scale 4 --is_PMG True --patch_size 384 --crop_piece 12 6 3 1 --stride 1 --loss_name 1_L1 --device cuda:1 >log/x4_MSRN_PMG_12_6_3_1_1_L1 2>&1 &
#32018

nohup python main.py --model_name RCAN --scale 4 --is_PMG False --patch_size 384 --loss_name 1_L1 --device cuda:0 >log/x4_RCAN_1_L1 2>&1 &
#
nohup python main.py --model_name RCAN --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 8 12 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_RCAN_PMG_1_3_6_8_12_1_L1 2>&1 &
#
nohup python main.py --model_name RCAN --scale 4 --is_PMG True --patch_size 384 --crop_piece 1 3 6 8 12 --stride 1 --loss_name 1_L1 --device cuda:0 >log/x4_RCAN_PMG_12_8_6_3_1_1_L1 2>&1 &
#
