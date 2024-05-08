import os
import scipy
import numpy as np
from torch.utils.data import Dataset
import torch

class SIDD_val(Dataset):
    '''
    SIDD validation dataset class 
    '''

    def __init__(self, opt):
        super().__init__()
        sidd_val_dir=opt['sidd_val_dir']
        len=opt['len']
        assert os.path.exists(
            sidd_val_dir), 'There is no dataset %s' % sidd_val_dir

        clean_mat_file_path = os.path.join(
            sidd_val_dir, 'ValidationGtBlocksSrgb.mat')
        noisy_mat_file_path = os.path.join(
            sidd_val_dir, 'ValidationNoisyBlocksSrgb.mat')

        self.clean_patches = np.array(scipy.io.loadmat(
            clean_mat_file_path, appendmat=False)['ValidationGtBlocksSrgb'])
        self.noisy_patches = np.array(scipy.io.loadmat(
            noisy_mat_file_path, appendmat=False)['ValidationNoisyBlocksSrgb'])
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, data_idx):
        img_id = data_idx // 32
        patch_id = data_idx % 32

        clean_img = self.clean_patches[img_id, patch_id, :].astype(float)
        noisy_img = self.noisy_patches[img_id, patch_id, :].astype(float)

        clean_img = torch.from_numpy(clean_img.transpose(2, 0, 1))
        noisy_img = torch.from_numpy(noisy_img.transpose(2, 0, 1))

        return {'clean': clean_img, 'noisy': noisy_img, 'clean_key': f'{img_id}_{patch_id}.png'}
