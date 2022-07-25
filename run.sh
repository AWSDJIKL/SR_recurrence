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

