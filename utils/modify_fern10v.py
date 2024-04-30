import numpy as np

scale_factor = 4

train_poses = np.load('data/split_allview_new/fern_10view_old/train_poses.npy')
train_poses[:, :, -1] *= (1 / scale_factor)

test_poses = np.load('data/split_allview_new/fern_10view_old/test_poses.npy')
test_poses[:, :, -1] *= (1 / scale_factor)

train_images = np.load('data/split_allview_new/fern_10view_old/train_images.npy')
train_images = train_images[:, ::scale_factor, ::scale_factor, :]

test_images = np.load('data/split_allview_new/fern_10view_old/test_images.npy')
test_images = test_images[:, ::scale_factor, ::scale_factor, :]

output_path = 'data/split_allview_new/fern_10view/'
np.save(output_path + 'train_poses.npy', train_poses)
np.save(output_path + 'test_poses.npy', test_poses)
np.save(output_path + 'train_images.npy', train_images)
np.save(output_path + 'test_images.npy', test_images)