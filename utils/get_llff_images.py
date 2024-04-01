import numpy as np
import cv2
import os
import argparse

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert images from npy to PNG')
    parser.add_argument('--project_dir', type=str, default='data/split_allview_new/fern_2view',
                        help='Directory of the project')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    project_dir = args.project_dir
    # Check if the image directory exists, if not, create it
    train_image_dir = os.path.join(project_dir, 'train_images')
    test_image_dir = os.path.join(project_dir, 'test_images')
    check_dir(train_image_dir)
    check_dir(test_image_dir)

    # Load the images from the npy file
    train_images = (np.load(os.path.join(project_dir, 'train_images.npy')) * 255).astype(np.uint8)

    # Save each image as PNG format
    for i, image in enumerate(train_images):
        image_path = os.path.join(train_image_dir, f'img_{str(i+1).zfill(4)}.png')
        cv2.imwrite(image_path, image)

    # Load the images from the npy file
    test_images = (np.load(os.path.join(project_dir, 'test_images.npy')) * 255).astype(np.uint8)

    # Save each image as PNG format
    for i, image in enumerate(test_images):
        image_path = os.path.join(test_image_dir, f'img_{str(i+1).zfill(4)}.png')
        cv2.imwrite(image_path, image)

    print(f'Done!')
