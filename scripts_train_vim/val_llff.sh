if [ "$IS_ON_MA" = "true" ]; then
    python_exe="/efs/y00855328/misc/miniforge3/envs/dsnerf/bin/python"
else
    python_exe="/home/ma-user/work/y00855328/misc/miniforge3/envs/dsnerf/bin/python"
fi

$python_exe run_nerf.py --config configs/fern_2v_vimcm.txt --basedir logs/output/fern --chunk 8192 --render_only --render_test $@
$python_exe run_nerf.py --config configs/fern_2v_vimcm.txt --datadir ./data/split_allview_new/fern_5view --expname fern_5v_vimcm --basedir logs/output/fern --chunk 8192 --render_only --render_test $@
# $python_exe run_nerf.py --config configs/fern_2v_vimcm.txt --datadir ./data/split_allview_new/fern_10view --expname fern_10v_vimcm --basedir logs/output/fern --chunk 8192 --render_only --render_test $@ 
$python_exe run_nerf.py --config configs/flower_2v_vimcm.txt --basedir logs/output/flower --chunk 8192 --render_only --render_test $@
$python_exe run_nerf.py --config configs/flower_2v_vimcm.txt --datadir ./data/split_allview_new/flower_5view --expname flower_5v_vimcm --basedir logs/output/flower --chunk 8192 --render_only --render_test $@
# $python_exe run_nerf.py --config configs/flower_2v_vimcm.txt --datadir ./data/split_allview_new/flower_10view --expname flower_10v_vimcm --basedir logs/output/flower --chunk 8192 --render_only --render_test $@
$python_exe run_nerf.py --config configs/fortress_2v_vimcm.txt --basedir logs/output/fortress --chunk 8192 --render_only --render_test $@
$python_exe run_nerf.py --config configs/fortress_2v_vimcm.txt --datadir ./data/split_allview_new/fortress_5view --expname fortress_5v_vimcm --basedir logs/output/fortress --chunk 8192 --render_only --render_test $@
# $python_exe run_nerf.py --config configs/fortress_2v_vimcm.txt --datadir ./data/split_allview_new/fortress_10view --expname fortress_10v_vimcm --basedir logs/output/fortress --chunk 8192 --render_only --render_test $@
$python_exe run_nerf.py --config configs/horns_2v_vimcm.txt --basedir logs/output/horns --chunk 8192 --render_only --render_test $@
$python_exe run_nerf.py --config configs/horns_2v_vimcm.txt --datadir ./data/split_allview_new/horns_5view --expname horns_5v_vimcm --basedir logs/output/horns --chunk 8192 --render_only --render_test $@
# $python_exe run_nerf.py --config configs/horns_2v_vimcm.txt --datadir ./data/split_allview_new/horns_10view --expname horns_10v_vimcm --basedir logs/output/horns --chunk 8192 --render_only --render_test $@
$python_exe run_nerf.py --config configs/leaves_2v_vimcm.txt --basedir logs/output/leaves --chunk 8192 --render_only --render_test $@
$python_exe run_nerf.py --config configs/leaves_2v_vimcm.txt --datadir ./data/split_allview_new/leaves_5view --expname leaves_5v_vimcm --basedir logs/output/leaves --chunk 8192 --render_only --render_test $@
# $python_exe run_nerf.py --config configs/leaves_2v_vimcm.txt --datadir ./data/split_allview_new/leaves_10view --expname leaves_10v_vimcm --basedir logs/output/leaves --chunk 8192 --render_only --render_test $@
$python_exe run_nerf.py --config configs/orchids_2v_vimcm.txt --basedir logs/output/orchids --chunk 8192 --render_only --render_test $@
$python_exe run_nerf.py --config configs/orchids_2v_vimcm.txt --datadir ./data/split_allview_new/orchids_5view --expname orchids_5v_vimcm --basedir logs/output/orchids --chunk 8192 --render_only --render_test $@
# $python_exe run_nerf.py --config configs/orchids_2v_vimcm.txt --datadir ./data/split_allview_new/orchids_10view --expname orchids_10v_vimcm --basedir logs/output/orchids --chunk 8192 --render_only --render_test $@
$python_exe run_nerf.py --config configs/room_2v_vimcm.txt --basedir logs/output/room --chunk 8192 --render_only --render_test $@
$python_exe run_nerf.py --config configs/room_2v_vimcm.txt --datadir ./data/split_allview_new/room_5view --expname room_5v_vimcm --basedir logs/output/room --chunk 8192 --render_only --render_test $@
# $python_exe run_nerf.py --config configs/room_2v_vimcm.txt --datadir ./data/split_allview_new/room_10view --expname room_10v_vimcm --basedir logs/output/room --chunk 8192 --render_only --render_test $@
$python_exe run_nerf.py --config configs/trex_2v_vimcm.txt --basedir logs/output/trex --chunk 8192 --render_only --render_test $@
$python_exe run_nerf.py --config configs/trex_2v_vimcm.txt --datadir ./data/split_allview_new/trex_5view --expname trex_5v_vimcm --basedir logs/output/trex --chunk 8192 --render_only --render_test $@
# $python_exe run_nerf.py --config configs/trex_2v_vimcm.txt --datadir ./data/split_allview_new/trex_10view --expname trex_10v_vimcm --basedir logs/output/trex --chunk 8192 --render_only --render_test $@
