import os
import pandas as pd
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

import numpy as np
import nibabel as nib
import tqdm

import monai.transforms as mtf
import pickle
from collections import defaultdict




def reshape_array(data, target_depth):
    _, _, n = data.shape

    if n > target_depth:
        start = (n - target_depth) // 2
        end = start + target_depth
        data = data[:, :, start:end]
    elif n < target_depth:
        padding = target_depth - n
        pad_width = ((0, 0), (0, 0), (0, padding))
        data = np.pad(data, pad_width, mode='constant', constant_values=0)
    else:
        data = data
    
    return data


class ARDSDataset(Dataset):
    def __init__(self, data_root, img_ids, num_classes, transform, image_size, mode, modal='mediastinum_window',use_norm=False):
        self.data_root = data_root
        self.label_file = os.path.join(data_root, 'label_v2.csv')
        self.img_ids = img_ids
        self.num_classes = num_classes
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.modal = modal

        self.load_data()

        idx = 0
        item = self.data_list[idx]
        data = item['data']
        label = item['label']
        data = data[np.newaxis, :]
        data = self.transform(data)

    def resize_volume(self, volume, target_shape):
        from scipy.ndimage import zoom
        factors = [t / s for s, t in zip(volume.shape, target_shape)]

        return zoom(volume, factors, order=1)

    def load_data(self):
        pkl_name = os.path.join(self.data_root, '{}_data_list.pkl'.format(self.mode))

        # if os.path.exists(pkl_name):
        #     with open(pkl_name, 'rb') as f:
        #         self.data_list = pickle.load(f)
        #     return

        self.data_list = []
        label_df = pd.read_csv(self.label_file)

        id_to_level = dict(zip(label_df.iloc[:, 0], label_df.iloc[:, 1]))
        tbar = tqdm.tqdm(total=len(self.img_ids))
        level_count = defaultdict(int)

        for img_id in self.img_ids:
            folder_path = os.path.join(self.data_root, img_id)
            img_id = int(img_id)
            if img_id not in id_to_level:
                continue

            level = id_to_level[img_id]

            for file_name in os.listdir(folder_path):
                if file_name.endswith('.nii.gz') and self.modal in file_name:
                    file_path = os.path.join(folder_path, file_name)
                    nii_data = nib.load(file_path)
                    data = nii_data.get_fdata()
                    
                    target_shape = (32, 256, 256)
                    data = np.transpose(data, (2, 0, 1))
                    data = self.resize_volume(data, target_shape)

                    data_dict = {
                        'data': data,
                        'label': level
                    }
                    level_count[level] += 1
                    self.data_list.append(data_dict)
            
            tbar.update(1)
        
        for level, count in level_count.items():
            print(f"Level {level}: {count} times")

        with open(pkl_name, 'wb') as f:
            pickle.dump(self.data_list, f)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        data = item['data']
        label = item['label']
        data = data[np.newaxis, :]
        data = self.transform(data)

        return data, label
