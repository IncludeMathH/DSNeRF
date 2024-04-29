if [ "$IS_ON_MA" = "true" ]; then
    python_exe="/efs/y00855328/misc/miniforge3/envs/dsnerf/bin/python"
else
    python_exe="/home/ma-user/work/y00855328/misc/miniforge3/envs/dsnerf/bin/python"
fi

modeltype="VimCM"

$python_exe run_nerf.py --config configs/fern_2v_vimcm.txt --datadir ./data/split_allview_new/fern_2view --expname fern_2v_${modeltype} --model_type ${modeltype} $@
$python_exe run_nerf.py --config configs/fern_2v_vimcm.txt --datadir ./data/split_allview_new/fern_5view --expname fern_5v_${modeltype} --model_type ${modeltype} $@
$python_exe run_nerf.py --config configs/fern_2v_vimcm.txt --datadir ./data/split_allview_new/fern_10view --expname fern_10v_${modeltype} --model_type ${modeltype} $@
