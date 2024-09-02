import os
import pandas as pd
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
import tqdm
import pickle
from collections import defaultdict
import torch

class SMOTE:
    def __init__(self, sampling_strategy='auto', k_neighbors=5):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # Flatten the 3D data into 2D
        n_samples, *original_shape = X.shape
        X_flattened = X.reshape(n_samples, -1)

        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]

        X_minority = X_flattened[y == minority_class]

        n_minority = len(X_minority)
        n_majority = len(X[y == majority_class])
        n_to_generate = (n_majority - n_minority)

        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors).fit(X_minority)
        indices = nbrs.kneighbors(X_minority, return_distance=False)

        synthetic_samples = []
        for i in range(n_to_generate):
            sample_idx = np.random.randint(0, n_minority)
            neighbor_idx = np.random.choice(indices[sample_idx])
            diff = X_minority[neighbor_idx] - X_minority[sample_idx]
            synthetic_sample = X_minority[sample_idx] + np.random.rand() * diff
            synthetic_samples.append(synthetic_sample)

        X_resampled = np.vstack((X_flattened, np.array(synthetic_samples)))
        y_resampled = np.hstack((y, np.full(n_to_generate, minority_class)))

        # Reshape the resampled data back to the original 3D shape
        X_resampled = X_resampled.reshape(-1, *original_shape)

        return torch.tensor(X_resampled, dtype=torch.float32), torch.tensor(y_resampled, dtype=torch.long)

class ARDSDataset(Dataset):
    def __init__(self, data_root, img_ids, num_classes, transform, image_size, mode, modal='mediastinum_window', use_norm=False, use_smote=False):
        self.data_root = data_root
        self.label_file = os.path.join(data_root, 'label_v2.csv')
        self.img_ids = img_ids
        self.num_classes = num_classes
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.modal = modal
        self.use_smote = use_smote

        self.load_data()

    def resize_volume(self, volume, target_shape):
        from scipy.ndimage import zoom
        factors = [t / s for s, t in zip(volume.shape, target_shape)]
        return zoom(volume, factors, order=1)

    def load_data(self):
        if self.use_smote:
            pkl_name = os.path.join(self.data_root, '{}_{}_data_list.pkl'.format(self.mode, self.modal))
        else:
            pkl_name = os.path.join(self.data_root, '{}_{}_smote_data_list.pkl'.format(self.mode, self.modal))

        # if os.path.exists(pkl_name):
        #     with open(pkl_name, 'rb') as f:
        #         self.data_list = pickle.load(f)
        #     return

        self.data_list = []
        label_df = pd.read_csv(self.label_file)

        id_to_level = dict(zip(label_df.iloc[:, 0], label_df.iloc[:, 1]))
        tbar = tqdm.tqdm(total=len(self.img_ids))
        level_count = defaultdict(int)

        features = []
        labels = []

        for img_id in self.img_ids:
            folder_path = os.path.join(self.data_root, img_id)
            img_id = int(img_id)
            if img_id not in id_to_level:
                continue

            level = id_to_level[img_id]
            if self.num_classes == 2 and level == 2:
                level = 1

            for file_name in os.listdir(folder_path):
                if file_name.endswith('.nii.gz') and self.modal in file_name:
                    file_path = os.path.join(folder_path, file_name)
                    nii_data = nib.load(file_path)
                    data = nii_data.get_fdata()
                    
                    target_shape = (32, 256, 256)
                    data = np.transpose(data, (2, 0, 1))
                    data = self.resize_volume(data, target_shape)

                    features.append(data)
                    labels.append(level)
                    level_count[level] += 1
            
            tbar.update(1)
        
        for level, count in level_count.items():
            print(f"Level {level}: {count} times")

        if self.use_smote and self.mode == 'train':
            # Apply SMOTE to balance the dataset
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(features, labels)

            new_level_count = defaultdict(int)
            for i in range(len(X_resampled)):
                data_dict = {
                    'data': X_resampled[i].numpy(),
                    'label': y_resampled[i].item()
                }
                self.data_list.append(data_dict)
                new_level_count[y_resampled[i].item()] += 1

            print("After SMOTE:")
            for level, count in new_level_count.items():
                print(f"Level {level}: {count} times")
        else:
            for i in range(len(features)):
                data_dict = {
                    'data': features[i],
                    'label': labels[i]
                }
                self.data_list.append(data_dict)

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
