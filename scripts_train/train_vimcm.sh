if [ "$IS_ON_MA" = "true" ]; then
    python_exe="/efs/y00855328/misc/miniforge3/envs/dsnerf/bin/python"
else
    python_exe="/home/ma-user/work/y00855328/misc/miniforge3/dsnerf/vim2/bin/python"
fi

nohup $python_exe run_nerf.py --config configs/fern_2v_vimcm.txt --device cuda:0 &
nohup $python_exe run_nerf.py --config configs/flower_2v_vimcm.txt --device cuda:1 &
nohup $python_exe run_nerf.py --config configs/fortress_2v_vimcm.txt --device cuda:2 &
nohup $python_exe run_nerf.py --config configs/horns_2v_vimcm.txt --device cuda:3 &
nohup $python_exe run_nerf.py --config configs/leaves_2v_vimcm.txt --device cuda:4 &
nohup $python_exe run_nerf.py --config configs/orchids_2v_vimcm.txt --device cuda:5 &
nohup $python_exe run_nerf.py --config configs/room_2v_vimcm.txt --device cuda:6 &
nohup $python_exe run_nerf.py --config configs/trex_2v_vimcm.txt --device cuda:7 &
