import numpy as np
import cv2, os, glob

num_view = '5'
for folder in glob.glob(f'data/split_allview_new/*{num_view}view'):
    for split in ['train', 'test']:
        images = np.load(f'{folder}/{split}_images.npy')
        base_cls = os.path.basename(folder).split('_')[0]
        out_dir = f'data/{split}_images/split_allview_new'
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        for i in range(images.shape[0]):
            image = images[i]
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            idx = str(i).zfill(3)
            cv2.imwrite(os.path.join(out_dir, f'{base_cls}_{num_view}v_{idx}.png'), image)
