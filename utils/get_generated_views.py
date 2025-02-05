import os
import glob
import shutil

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# 找到所有以'Loss'结尾的文件夹
dirs = [d for d in glob.glob('logs/baseline/*mamba_v1') if os.path.isdir(d)]
output_dir = 'logs/baseline_mamba_v1/generated_views'
check_dir(output_dir)

for dir in dirs:
    # 找到最新的.png文件
    list_of_files = glob.glob(f'{dir}/testset_200000/*.png')

    # 复制文件到新的位置
    for file in list_of_files:
        if not file.endswith('depth.png'):
            # 复制文件到新的位置
            tgt_name = '_'.join(file.split('/')[-3].split('_')[:2]) + '_' + file.split('/')[-1]
            shutil.copy(file, os.path.join(output_dir, tgt_name))

# list_of_files = glob.glob(f'logs/mambav2/flower_2v_mamba_RelativeLoss/testset_050000/*.png')
# # 复制文件到新的位置
# for file in list_of_files:
#     if not file.endswith('depth.png'):
#         # 复制文件到新的位置
#         tgt_name = '_'.join(file.split('/')[-3].split('_')[:2]) + '_' + file.split('/')[-1]
#         shutil.copy(file, os.path.join(output_dir, tgt_name))
