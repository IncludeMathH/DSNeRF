modeltype="Vim"
device=$1

python run_nerf.py --config configs/room_2v_vimcm.txt --datadir ./data/split_allview_new/room_2view --expname room_2v_DSNeRF${modeltype} --model_type ${modeltype} --colmap_depth --depth_loss --depth_lambda 0.1 --weighted_loss --N_rand 4096 --chunk 2048 --netchunk 4096 --N_iters 50000 --lrate 5e-4 --i_testset 50000 --device $device --use_wandb
python run_nerf.py --config configs/room_2v_vimcm.txt --datadir ./data/split_allview_new/room_5view --expname room_5v_DSNeRF${modeltype} --model_type ${modeltype} --colmap_depth --depth_loss --depth_lambda 0.1 --weighted_loss --N_rand 4096 --chunk 2048 --netchunk 4096 --N_iters 50000 --lrate 5e-4 --i_testset 50000 --device $device --use_wandb
python run_nerf.py --config configs/room_2v_vimcm.txt --datadir ./data/split_allview_new/room_10view --expname room_10v_DSNeRF${modeltype} --model_type ${modeltype} --colmap_depth --depth_loss --depth_lambda 0.1 --weighted_loss --N_rand 4096 --chunk 2048 --netchunk 4096 --N_iters 50000 --lrate 5e-4 --i_testset 50000 --device $device --use_wandb
