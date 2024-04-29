if [ "$IS_ON_MA" = "true" ]; then
    python_exe="/efs/y00855328/misc/miniforge3/envs/dsnerf/bin/python"
else
    python_exe="/home/ma-user/work/y00855328/misc/miniforge3/envs/dsnerf/bin/python"
fi

modeltype="Vim"

$python_exe run_nerf.py --config configs/horns_2v_vimcm.txt --datadir ./data/split_allview_new/horns_2view --expname horns_2v_DSNeRF${modeltype} --model_type ${modeltype} --colmap_depth --depth_loss --depth_lambda 0.1 --weighted_loss $@
$python_exe run_nerf.py --config configs/horns_2v_vimcm.txt --datadir ./data/split_allview_new/horns_5view --expname horns_5v_DSNeRF${modeltype} --model_type ${modeltype} --colmap_depth --depth_loss --depth_lambda 0.1 --weighted_loss $@
$python_exe run_nerf.py --config configs/horns_2v_vimcm.txt --datadir ./data/split_allview_new/horns_10view --expname horns_10v_DSNeRF${modeltype} --model_type ${modeltype} --colmap_depth --depth_loss --depth_lambda 0.1 --weighted_loss $@
