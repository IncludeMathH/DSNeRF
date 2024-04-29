if [ "$IS_ON_MA" = "true" ]; then
    python_exe="/efs/y00855328/misc/miniforge3/envs/dsnerf/bin/python"
else
    python_exe="/home/ma-user/work/y00855328/misc/miniforge3/envs/dsnerf/bin/python"
fi

modeltype="CA_Vim"

$python_exe run_nerf.py --config configs/leaves_2v_vimcm.txt --datadir ./data/split_allview_new/leaves_2view --expname leaves_2v_${modeltype} --model_type ${modeltype} $@
$python_exe run_nerf.py --config configs/leaves_2v_vimcm.txt --datadir ./data/split_allview_new/leaves_5view --expname leaves_5v_${modeltype} --model_type ${modeltype} $@
$python_exe run_nerf.py --config configs/leaves_2v_vimcm.txt --datadir ./data/split_allview_new/leaves_10view --expname leaves_10v_${modeltype} --model_type ${modeltype} $@
