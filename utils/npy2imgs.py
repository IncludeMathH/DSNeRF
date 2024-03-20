import numpy as np
import cv2, os

# 加载.npy文件
images = np.load('/data3/dn/project/DSNeRF/data/split_allview_new/fern_2view/train_images.npy')
out_dir = '/data3/dn/project/DSNeRF/data/split_allview_new/images/fren_2view'

# 创建一个新的文件夹来保存图像
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# 遍历所有图像
for i in range(images.shape[0]):
    # 获取一张图像
    image = images[i]

    # 将图像数据转换为8位整数
    image = (image * 255).astype(np.uint8)

    # 保存图像为.png文件
    cv2.imwrite(os.path.join(out_dir, f'image{i}.png'), image)
