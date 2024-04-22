if [ "$IS_ON_MA" = "true" ]; then
    python_exe="/efs/y00855328/misc/miniforge3/envs/dsnerf/bin/python"
else
    python_exe="/home/ma-user/work/y00855328/misc/miniforge3/dsnerf/vim2/bin/python"
fi

nohup $python_exe run_nerf.py --config configs/fern_2v_vimcm.txt --device cuda:0 --datadir ./data/split_allview_new/fern_10view --expname fern_10v_vimcm &
nohup $python_exe run_nerf.py --config configs/flower_2v_vimcm.txt --device cuda:1 --datadir ./data/split_allview_new/flower_10view --expname flower_10v_vimcm &
nohup $python_exe run_nerf.py --config configs/fortress_2v_vimcm.txt --device cuda:2 --datadir ./data/split_allview_new/fortress_10view --expname fortress_10v_vimcm &
nohup $python_exe run_nerf.py --config configs/horns_2v_vimcm.txt --device cuda:3 --datadir ./data/split_allview_new/horns_10view --expname horns_10v_vimcm &
nohup $python_exe run_nerf.py --config configs/leaves_2v_vimcm.txt --device cuda:4 --datadir ./data/split_allview_new/leaves_10view --expname leaves_10v_vimcm &
nohup $python_exe run_nerf.py --config configs/orchids_2v_vimcm.txt --device cuda:5 --datadir ./data/split_allview_new/orchids_10view --expname orchids_10v_vimcm &
nohup $python_exe run_nerf.py --config configs/room_2v_vimcm.txt --device cuda:6 --datadir ./data/split_allview_new/room_10view --expname room_10v_vimcm &
nohup $python_exe run_nerf.py --config configs/trex_2v_vimcm.txt --device cuda:7 --datadir ./data/split_allview_new/trex_10view --expname trex_10v_vimcm &
