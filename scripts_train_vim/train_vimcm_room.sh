modeltype="Vim"
device=$1

python run_nerf.py --config configs/room_2v_vimcm.txt --datadir ./data/split_allview_new/room_2view --expname room_2v_${modeltype} --model_type ${modeltype} --N_rand 1024 --lrate 1.25e-4 --N_iters 200000 --device ${device} --chunk 1024 --netchunk 2048 --use_wandb
python run_nerf.py --config configs/room_2v_vimcm.txt --datadir ./data/split_allview_new/room_5view --expname room_5v_${modeltype} --model_type ${modeltype} --N_rand 1024 --lrate 1.25e-4 --N_iters 200000 --device ${device} --chunk 1024 --netchunk 2048 --use_wandb
python run_nerf.py --config configs/room_2v_vimcm.txt --datadir ./data/split_allview_new/room_10view --expname room_10v_${modeltype} --model_type ${modeltype} --N_rand 1024 --lrate 1.25e-4 --N_iters 200000 --device ${device} --chunk 1024 --netchunk 2048 --use_wandb
