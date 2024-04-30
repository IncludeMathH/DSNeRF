modeltype="Vim"
device=$1

python run_nerf.py --config configs/horns_2v_vimcm.txt --datadir ./data/split_allview_new/horns_2view --expname horns_2v_${modeltype} --model_type ${modeltype} --N_rand 1024 --lrate 1.25e-4 --N_iters 200000 --device ${device} --chunk 1024 --netchunk 2048 --use_wandb
python run_nerf.py --config configs/horns_2v_vimcm.txt --datadir ./data/split_allview_new/horns_5view --expname horns_5v_${modeltype} --model_type ${modeltype} --N_rand 1024 --lrate 1.25e-4 --N_iters 200000 --device ${device} --chunk 1024 --netchunk 2048 --use_wandb
python run_nerf.py --config configs/horns_2v_vimcm.txt --datadir ./data/split_allview_new/horns_10view --expname horns_10v_${modeltype} --model_type ${modeltype} --N_rand 1024 --lrate 1.25e-4 --N_iters 200000 --device ${device} --chunk 1024 --netchunk 2048 --use_wandb
