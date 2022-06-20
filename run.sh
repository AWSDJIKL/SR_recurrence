python prepare_datasets.py
python main.py --model_name SRCNN --scale 2
python main.py --model_name SRCNN --scale 3
python main.py --model_name SRCNN --scale 4
python main.py --model_name SRCNN --scale 8

python main.py --model_name FSRCNN --scale 2
python main.py --model_name FSRCNN --scale 3
python main.py --model_name FSRCNN --scale 4
python main.py --model_name FSRCNN --scale 8

python main.py --model_name ESPCN --scale 2
python main.py --model_name ESPCN --scale 3
python main.py --model_name ESPCN --scale 4
python main.py --model_name ESPCN --scale 8

python main.py --model_name RCAN --scale 2 --n_resgroups 10 --n_resblocks 20
python main.py --model_name RCAN --scale 3 --n_resgroups 10 --n_resblocks 20
python main.py --model_name RCAN --scale 4 --n_resgroups 10 --n_resblocks 20
python main.py --model_name RCAN --scale 8 --n_resgroups 10 --n_resblocks 20

python main.py --model_name MSRN --scale 2
python main.py --model_name MSRN --scale 3
python main.py --model_name MSRN --scale 4
python main.py --model_name MSRN --scale 8

python main.py --model_name MSARN --scale 2 --is_PMG False --loss_name 1_L1
python main.py --model_name MSARN --scale 3 --is_PMG False --loss_name 1_L1
python main.py --model_name MSARN --scale 4 --is_PMG False --loss_name 1_L1
python main.py --model_name MSARN --scale 8 --is_PMG False --loss_name 1_L1

python main.py --model_name MSARN --scale 2 --is_PMG False --loss_name 1_L1+1e-3_VGG
python main.py --model_name MSARN --scale 3 --is_PMG False --loss_name 1_L1+1e-3_VGG
python main.py --model_name MSARN --scale 4 --is_PMG False --loss_name 1_L1+1e-3_VGG
python main.py --model_name MSARN --scale 8 --is_PMG False --loss_name 1_L1+1e-3_VGG

python main.py --model_name MSARN --scale 2 --is_PMG True --loss_name 1_L1
python main.py --model_name MSARN --scale 3 --is_PMG True --loss_name 1_L1
python main.py --model_name MSARN --scale 4 --is_PMG True --loss_name 1_L1
python main.py --model_name MSARN --scale 8 --is_PMG True --loss_name 1_L1

python main.py --model_name MSARN --scale 2 --is_PMG True --loss_name 1_L1+1e-3_VGG
python main.py --model_name MSARN --scale 3 --is_PMG True --loss_name 1_L1+1e-3_VGG
python main.py --model_name MSARN --scale 4 --is_PMG True --loss_name 1_L1+1e-3_VGG
python main.py --model_name MSARN --scale 8 --is_PMG True --loss_name 1_L1+1e-3_VGG
