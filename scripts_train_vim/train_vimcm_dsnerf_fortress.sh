modeltype="Vim"
device=$1

python run_nerf.py --config configs/fortress_2v_vimcm.txt --datadir ./data/split_allview_new/fortress_2view --expname fortress_2v_DSNeRF${modeltype} --model_type ${modeltype} --colmap_depth --depth_loss --depth_lambda 0.1 --weighted_loss --N_rand 4096 --chunk 2048 --netchunk 4096 --N_iters 50000 --lrate 5e-4 --i_testset 50000 --device $device --use_wandb
python run_nerf.py --config configs/fortress_2v_vimcm.txt --datadir ./data/split_allview_new/fortress_5view --expname fortress_5v_DSNeRF${modeltype} --model_type ${modeltype} --colmap_depth --depth_loss --depth_lambda 0.1 --weighted_loss --N_rand 4096 --chunk 2048 --netchunk 4096 --N_iters 50000 --lrate 5e-4 --i_testset 50000 --device $device --use_wandb
python run_nerf.py --config configs/fortress_2v_vimcm.txt --datadir ./data/split_allview_new/fortress_10view --expname fortress_10v_DSNeRF${modeltype} --model_type ${modeltype} --colmap_depth --depth_loss --depth_lambda 0.1 --weighted_loss --N_rand 4096 --chunk 2048 --netchunk 4096 --N_iters 50000 --lrate 5e-4 --i_testset 50000 --device $device --use_wandb
