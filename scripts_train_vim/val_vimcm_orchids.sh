modeltype="Vim"
device=$1

# python run_nerf.py --config configs/orchids_2v_vimcm.txt --datadir ./data/split_allview_new/orchids_2view --expname orchids_2v_${modeltype} --model_type ${modeltype} --N_rand 1024 --lrate 1.25e-4 --N_iters 200000 --device ${device} --chunk 1024 --netchunk 2048 --render_only --render_test
# python run_nerf.py --config configs/orchids_2v_vimcm.txt --datadir ./data/split_allview_new/orchids_5view --expname orchids_5v_${modeltype} --model_type ${modeltype} --N_rand 1024 --lrate 1.25e-4 --N_iters 200000 --device ${device} --chunk 1024 --netchunk 2048 --render_only --render_test
python run_nerf.py --config configs/orchids_2v_vimcm.txt --datadir ./data/split_allview_new/orchids_10view --expname orchids_10v_${modeltype} --model_type ${modeltype} --N_rand 1024 --lrate 1.25e-4 --N_iters 200000 --device ${device} --chunk 1024 --netchunk 2048 --render_only --render_test
