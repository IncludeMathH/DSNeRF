import os
import glob
import shutil
import numpy as np
import cv2

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# 找到所有以'Loss'结尾的文件夹
dirs = [d for d in glob.glob('data/split_allview_new/*2view') if os.path.isdir(d)]
output_dir = 'data/split_allview_new_2view_test_images'
check_dir(output_dir)

for dir in dirs:
    # Load the images from the npy file
    test_images = (np.load(os.path.join(dir, 'test_images.npy')) * 255).astype(np.uint8)

    # Save each image as PNG format
    for i, image in enumerate(test_images):
        image_path = os.path.join(output_dir, '_'.join(dir.split('/')[-1].split('_')[:2]).replace('view', 'v') + f'_{str(i).zfill(3)}.png')
        cv2.imwrite(image_path, image)
