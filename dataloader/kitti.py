import os
import cv2
import numpy as np
import torch.utils.data
from glob import glob
import os.path as osp
from utils.cam_utils import load_calib

from .transforms import ProcessData
class KITTI_hplflownet(torch.utils.data.Dataset): # hplflownet
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """

    def __init__(self,
                 transform=None,
                 train=False,
                 num_points=8192,
                 data_root='datasets/datasets_KITTI_hplflownet/',
                 remove_ground = True):
        self.root = osp.join(data_root, 'KITTI_processed_occ_final')
        #assert train is False
        self.train = train
        self.transform = ProcessData(data_process_args = {'DEPTH_THRESHOLD':35., 'NO_CORR':True},
                                    num_points=8192,
                                    allow_less_points=False)
        self.num_points = num_points
        self.remove_ground = remove_ground

        self.samples = self.make_dataset()
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data_dict = {'index': index}

        proj_mat = load_calib(os.path.join('datasets/KITTI_stereo2015/training/', 'calib_cam_to_cam', '%06d.txt' % index))
        f, cx, cy = proj_mat[0, 0], proj_mat[0, 2], proj_mat[1, 2]

        pc1_loaded, pc2_loaded = self.pc_loader(self.samples[index])
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        pcs = np.concatenate([pc1_transformed, pc2_transformed], axis=1)

        pcs = torch.from_numpy(pcs).permute(0, 1).float()
        flow_3d = torch.from_numpy(sf_transformed).permute(1, 0).float()
        intrinsics = torch.from_numpy(np.float32([f, cx, cy]))

        data_dict['input_h'] = 384
        data_dict['input_w'] = 1248
        data_dict['pcs'] = pcs
        data_dict['flow_3d'] = flow_3d
        data_dict['intrinsics'] = intrinsics
        return data_dict

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str

    def make_dataset(self):
        do_mapping = True
        root = osp.realpath(osp.expanduser(self.root))

        all_paths = sorted(os.walk(root))
        useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]
        try:
            assert (len(useful_paths) == 200)
        except AssertionError:
            print('assert (len(useful_paths) == 200) failed!', len(useful_paths))

        if do_mapping:
            mapping_path = osp.join(osp.dirname(__file__), 'KITTI_mapping.txt')
            print('mapping_path', mapping_path)

            with open(mapping_path) as fd:
                lines = fd.readlines()
                lines = [line.strip() for line in lines]
            useful_paths = [path for path in useful_paths if lines[int(osp.split(path)[-1])] != '']

        res_paths = useful_paths

        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        pc1 = np.load(osp.join(path, 'pc1.npy'))  #.astype(np.float32)
        pc2 = np.load(osp.join(path, 'pc2.npy'))  #.astype(np.float32)

        if self.remove_ground:
            is_ground = np.logical_and(pc1[:,1] < -1.4, pc2[:,1] < -1.4)
            not_ground = np.logical_not(is_ground)

            pc1 = pc1[not_ground]
            pc2 = pc2[not_ground]

        return pc1, pc2

class KITTI_flownet3d(torch.utils.data.Dataset): # flownet3d
    def __init__(self, split='training40', root='datasets/KITTI_stereo2015'):
        assert os.path.isdir(root)
        assert split in ['training200', 'training160', 'training150', 'training40']

        self.root_dir = os.path.join(root, 'training')
        self.split = split
        self.augmentation = False
        self.n_points = 8192# 8192
        self.max_depth = 30

        if self.split == 'training200':
            self.indices = np.arange(200)
        if self.split == 'training150':
            self.indices = np.arange(150)
        elif self.split == 'training160':
            self.indices = [i for i in range(200) if i % 5 != 0]
        elif self.split == 'training40':
            self.indices = [i for i in range(200) if i % 5 == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if not self.augmentation:
            np.random.seed(23333)

        index = self.indices[i]
        data_dict = {'index': index}

        proj_mat = load_calib(os.path.join(self.root_dir, 'calib_cam_to_cam', '%06d.txt' % index))
        f, cx, cy = proj_mat[0, 0], proj_mat[0, 2], proj_mat[1, 2]


        path = os.path.join('datasets/datasets_KITTI_flownet3d/kitti_rm_ground/', '%06d.npz' % index)
        data = np.load(path)
        pc1 = np.concatenate((data['pos1'][:,1:2], data['pos1'][:,2:3], data['pos1'][:,0:1]), axis=1)
        pc2 = np.concatenate((data['pos2'][:,1:2], data['pos2'][:,2:3], data['pos2'][:,0:1]), axis=1)
        # limit max depth
        #pc1 = pc1[pc1[..., -1] < self.max_depth]
        #pc2 = pc2[pc2[..., -1] < self.max_depth]
        flow_3d = np.concatenate((data['gt'][:,1:2], data['gt'][:,2:3], data['gt'][:,0:1]), axis=1)

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=self.n_points, replace=pc1.shape[0] < self.n_points)
        indices2 = np.random.choice(pc2.shape[0], size=self.n_points, replace=pc2.shape[0] < self.n_points)
        pc1, pc2, flow_3d = pc1[indices1], pc2[indices2], flow_3d[indices1]

        pcs = np.concatenate([pc1, pc2], axis=1)

        pcs = torch.from_numpy(pcs).permute(0, 1).float()
        flow_3d = torch.from_numpy(flow_3d).permute(1, 0).float()
        intrinsics = torch.from_numpy(np.float32([f, cx, cy]))

        # images from KITTI have various sizes, padding them to a unified size of 1242x376
        data_dict['input_h'] = 384
        data_dict['input_w'] = 1248
        data_dict['pcs'] = pcs
        data_dict['flow_3d'] = flow_3d
        data_dict['intrinsics'] = intrinsics

        return data_dict