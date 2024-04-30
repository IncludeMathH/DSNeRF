modeltype="Vim"
device=$1

python run_nerf.py --config configs/fortress_2v_vimcm.txt --datadir ./data/split_allview_new/fortress_2view --expname fortress_2v_${modeltype} --model_type ${modeltype} --N_rand 1024 --lrate 1.25e-4 --N_iters 200000 --device ${device} --chunk 1024 --netchunk 2048 --use_wandb
python run_nerf.py --config configs/fortress_2v_vimcm.txt --datadir ./data/split_allview_new/fortress_5view --expname fortress_5v_${modeltype} --model_type ${modeltype} --N_rand 1024 --lrate 1.25e-4 --N_iters 200000 --device ${device} --chunk 1024 --netchunk 2048 --use_wandb
python run_nerf.py --config configs/fortress_2v_vimcm.txt --datadir ./data/split_allview_new/fortress_10view --expname fortress_10v_${modeltype} --model_type ${modeltype} --N_rand 1024 --lrate 1.25e-4 --N_iters 200000 --device ${device} --chunk 1024 --netchunk 2048 --use_wandb
