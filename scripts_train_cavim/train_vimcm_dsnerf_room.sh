if [ "$IS_ON_MA" = "true" ]; then
    python_exe="/efs/y00855328/misc/miniforge3/envs/dsnerf/bin/python"
else
    python_exe="/home/ma-user/work/y00855328/misc/miniforge3/envs/dsnerf/bin/python"
fi

modeltype="CA_Vim"

$python_exe run_nerf.py --config configs/room_2v_vimcm.txt --datadir ./data/split_allview_new/room_2view --expname room_2v_DSNeRF${modeltype} --model_type ${modeltype} --colmap_depth --depth_loss --depth_lambda 0.1 --weighted_loss $@
$python_exe run_nerf.py --config configs/room_2v_vimcm.txt --datadir ./data/split_allview_new/room_5view --expname room_5v_DSNeRF${modeltype} --model_type ${modeltype} --colmap_depth --depth_loss --depth_lambda 0.1 --weighted_loss $@
$python_exe run_nerf.py --config configs/room_2v_vimcm.txt --datadir ./data/split_allview_new/room_10view --expname room_10v_DSNeRF${modeltype} --model_type ${modeltype} --colmap_depth --depth_loss --depth_lambda 0.1 --weighted_loss $@
