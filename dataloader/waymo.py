import os
import cv2
import numpy as np
import torch.utils.data
from glob import glob
import os.path as osp
from pathlib import Path
from utils.cam_utils import load_calib

from .transforms import ProcessData

class Waymo(torch.utils.data.Dataset): # flownet3d
    def __init__(self, split='train', root='datasets/waymo-open/train_extract'):
        assert os.path.isdir(root)

        self.root_dir = root
        self.split = split
        self.augmentation = False
        self.n_points = 8192# 8192
        self.max_depth = 30

        if self.split == 'train':
            imageset_path = str(Path(self.root_dir) / 'ImageSets' / (split.lower() + '.txt'))
            self.indices = []
            with open(imageset_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                #self.indices.append(format(line, '07d'))
                scene_id = format(int(line), '07d')[1:4]
                path = 'datasets/waymo-open/train_extract/' + str(scene_id) + '/sf_data/' + '%07d.npz' % int(line)
                if os.path.isfile(path):
                    self.indices.append(int(line))
            #self.indices = np.arange(0, 197)
            #self.scene_id_list = np.arange(100)
            #self.indices = self.indices[:19803]
        if self.split == 'valid':
            self.root_dir = 'datasets/waymo-open/valid_extract'
            imageset_path = str(Path(self.root_dir) / 'ImageSets' / (split.lower() + '.txt'))
            self.indices = []
            with open(imageset_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                #self.indices.append(format(line, '07d'))
                scene_id = format(int(line), '07d')[1:4]
                path = 'datasets/waymo-open/valid_extract/' + str(scene_id) + '/sf_data/' + '%07d.npz' % int(line[1:])
                if os.path.isfile(path):
                    self.indices.append(int(line))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if not self.augmentation:
            np.random.seed(23333)

        index = self.indices[i]
        data_dict = {'index': index}

        # camera intrinsics
        #f, cx, cy = 1050, 479.5, 269.5

        #proj_mat = load_calib(os.path.join(self.root_dir, 'calib_cam_to_cam', '%06d.txt' % index))
        #f, cx, cy = proj_mat[0, 0], proj_mat[0, 2], proj_mat[1, 2]
        scene_id = format(index, '07d')[1:4]
        index = int(format(index, '07d')[1:])
        if self.split == 'train':
            path = 'datasets/waymo-open/train_extract/' + str(scene_id) + '/sf_data/' + '%07d.npz' % index
        if self.split == 'valid':
            path = 'datasets/waymo-open/valid_extract/' + str(scene_id) + '/sf_data/' + '%07d.npz' % index
        #path = os.path.join('datasets/waymo_scene_flow/created/', str(scene_id), '/sf_data/', '%07d.npz' % index)
        #print(path)
        #path = os.path.join('datasets/waymo_scene_flow/created/000/sf_data/', '%07d.npz' % index)
        data = np.load(path)
        proj_mat = data['proj_mat']
        f, cx, cy = proj_mat[0, 0], proj_mat[0, 2], proj_mat[1, 2]
        pc1 = np.concatenate((data['pc1'][:,1:2], data['pc1'][:,2:3], data['pc1'][:,0:1]), axis=1)
        pc2 = np.concatenate((data['pc2'][:,1:2], data['pc2'][:,2:3], data['pc2'][:,0:1]), axis=1)
        #print(torch.mean(torch.tensor(pc1), dim=0))
        #print(torch.mean(torch.tensor(pc2), dim=0))
        # limit max depth
        pc1 = pc1[pc1[..., -1] < self.max_depth]
        pc2 = pc2[pc2[..., -1] < self.max_depth]
        flow_3d = np.concatenate((data['gt'][:,1:2], data['gt'][:,2:3], data['gt'][:,0:1]), axis=1)
        #print(torch.mean(torch.tensor(flow_3d), dim=0))
        #flow_3d = data['gt']

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=self.n_points, replace=pc1.shape[0] < self.n_points)
        indices2 = np.random.choice(pc2.shape[0], size=self.n_points, replace=pc2.shape[0] < self.n_points)
        pc1, pc2, flow_3d = pc1[indices1], pc2[indices2], flow_3d[indices1]

        pcs = np.concatenate([pc1, pc2], axis=1)

        pcs = torch.from_numpy(pcs).permute(0, 1).float()
        flow_3d = torch.from_numpy(flow_3d).permute(1, 0).float()
        intrinsics = torch.from_numpy(np.float32([f, cx, cy]))

        # images from KITTI have various sizes, padding them to a unified size of 1242x376
        data_dict['input_h'] = 1280
        data_dict['input_w'] = 1920
        data_dict['pcs'] = pcs
        data_dict['flow_3d'] = flow_3d
        data_dict['intrinsics'] = intrinsics

        return data_dict