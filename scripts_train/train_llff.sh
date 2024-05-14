modeltype=$1
view_type=$2
view_num=$3
device=$4


if [ "$modeltype" = "Vim" ]; then
    echo "using Vim model"
elif [ "$modeltype" = "NeRF" ]; then
    echo "using NeRF"
else
    echo "Invalid model type"
fi

python run_nerf.py --config configs/${view_type}_2v_vimcm.txt --basedir ./logs/${modeltype} --datadir ./data/split_allview_new/${view_type}_${view_num}view --expname ${view_type}_${view_num}v_${modeltype} --model_type ${modeltype} --N_rand 3072 --chunk 2048 --netchunk 4096 --N_iters 70000 --lrate 3.75e-4 --i_testset 70000 --device ${device} --use_wandb