if [ "$IS_ON_MA" = "true" ]; then
    python_exe="/efs/y00855328/misc/miniforge3/envs/dsnerf/bin/python"
else
    python_exe="/home/ma-user/work/y00855328/misc/miniforge3/envs/dsnerf/bin/python"
fi

modeltype="VimCm"
device=$1

python run_nerf.py --config configs/room_2v_vimcm.txt --datadir ./data/split_allview_new/room_2view --expname room_2v_${modeltype} --model_type ${modeltype} --device ${device} --use_wandb
# python run_nerf.py --config configs/room_2v_vimcm.txt --datadir ./data/split_allview_new/room_5view --expname room_5v_${modeltype} --model_type ${modeltype} --device ${device} --use_wandb
# python run_nerf.py --config configs/room_2v_vimcm.txt --datadir ./data/split_allview_new/room_10view --expname room_10v_${modeltype} --model_type ${modeltype} --device ${device} --use_wandb
