modeltype="Vim"
device=$1

# python run_nerf.py --config configs/leaves_2v_vimcm.txt --datadir ./data/split_allview_new/leaves_2view --expname leaves_2v_${modeltype} --model_type ${modeltype} --N_rand 1024 --lrate 1.25e-4 --N_iters 200000 --device ${device} --chunk 1024 --netchunk 2048 --render_only --render_test
# python run_nerf.py --config configs/leaves_2v_vimcm.txt --datadir ./data/split_allview_new/leaves_5view --expname leaves_5v_${modeltype} --model_type ${modeltype} --N_rand 1024 --lrate 1.25e-4 --N_iters 200000 --device ${device} --chunk 1024 --netchunk 2048 --render_only --render_test
python run_nerf.py --config configs/leaves_2v_vimcm.txt --datadir ./data/split_allview_new/leaves_10view --expname leaves_10v_${modeltype} --model_type ${modeltype} --N_rand 1024 --lrate 1.25e-4 --N_iters 200000 --device ${device} --chunk 1024 --netchunk 2048 --render_only --render_test
