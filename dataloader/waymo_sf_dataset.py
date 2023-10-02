from pathlib import Path
from concurrent import futures as futures
import random
from tqdm import tqdm

import numpy as np
from torch.utils.data import Dataset

from dataloader.waymo_utils import box_np_ops
from dataloader.waymo_utils.waymo_utils import get_waymo_frame_info
from dataloader.waymo_utils.crop_ground import CropGroundAuto


class WaymoSFDataset(Dataset):
    """Waymo scene flow dataset.
    TODO: Add online mode to process data during training to save storage space.
    """
    def __init__(self,
                 data_root):
        super().__init__()
        self.data_root = data_root
    
    def creat_data(self,
                   data_path=None, 
                   save_path=None, 
                   scene_id_list=[], 
                   split='train',
                   pc_num_features=8, # 8 -- waymo > 1.3, 6 -- waymo 1.2
                   rm_ground=True,
                   crop_params_save_path=None,
                   crop_params_load_path=None,
                   num_workers=8):
        """Create info file of waymo dataset. Given the data created by mmdetection3d, 
        generate scene flow data.

        Args:
            data_path (str): Path of the data root.
            save_path (str | None): Path to save the info file.
            scene_id_list (List[int]): List of the scene id to create scene flow.
            split (str, optional): Train, val or other split of the dataset.
            relative_path (bool, optional)
            pc_num_features (int): 8 or 6 for waymo
            num_workers (int)
        """
        if data_path is None:
            data_path = self.data_root
        if save_path is None:
            save_path = Path(data_path)
        else:
            save_path = Path(save_path)
        frame_idx = self.get_frame_idx(self.data_root, scene_id_list, split=split)
        frame_infos_dict = get_waymo_frame_info(
            data_path,
            label_info=True,
            velodyne=True,
            pc_num_features=pc_num_features,
            calib=True,
            pose=True,
            image=False,
            with_imageshape=False,
            extend_matrix=True,
            frame_idx=frame_idx,
            num_workers=num_workers,
            relative_path=True)
        if rm_ground:
            croper = CropGroundAuto()
            crop_pointer_dict = self.generate_crop_params(
                croper=croper,
                frame_infos_dict=frame_infos_dict, 
                scene_id_list=scene_id_list, 
                frame_idx=frame_idx, 
                data_path=data_path, 
                split=split, 
                pc_num_features=pc_num_features,
                save_path=crop_params_save_path,
                load_path=crop_params_load_path)
        else:
            croper, crop_pointer_dict = None, None
        self.generate_scene_flow(
            data_path,
            save_path,
            frame_idx,
            frame_infos_dict,
            remove_outside=False,
            pc_num_features=pc_num_features,
            croper=croper,
            crop_pointer_dict=crop_pointer_dict,
            downsample=None,
            num_workers=num_workers)
        print('Finish.')
    
    def generate_crop_params(self,
                             croper,
                             frame_infos_dict, 
                             scene_id_list, 
                             frame_idx, 
                             data_path, 
                             split, 
                             pc_num_features,
                             save_path=None,
                             load_path=None):
        """Generate crop parameters for remove ground.
        """
        if load_path is not None:
            crop_pointer_dict = croper.load_crop_pointer_dict(load_path)
            print('Load {} sceces parameters of croping ground from '.format(
                len(crop_pointer_dict.keys())))
            cur_s_id_list = [_ for _ in scene_id_list if _ not in crop_pointer_dict.keys()]
        else:
            crop_pointer_dict = {}
            cur_s_id_list = scene_id_list
        if cur_s_id_list != []:
            assert pc_num_features == 8, 'Need segmentation label.'
            print('Start to compute parameters of croping ground.')
            f_idx_array = np.array(frame_idx)
            s_idx_array = (f_idx_array // 1000).astype(np.int) % 1000
            #perfix = 0 if split.lower() == 'train' else 1
            perfix = 0 if split.lower() == 'train' else 0
            try:
                for s_id in tqdm(cur_s_id_list, total=len(cur_s_id_list), ncols=100):
                    s_f_id = f_idx_array[s_idx_array == s_id]
                    # f_num = int(s_f_id.max()) + 1
                    f_num = len(s_f_id)
                    # assert f_num == len(s_f_id)

                    crop_pointer_scene = croper.gen_crop_pointer_scene(f_num)
                    croper.reset_eval_thresh()
                    label_ind = np.zeros(f_num, dtype=np.bool)
                    record_pointer = crop_pointer_scene[24]
                    for f_id in range(f_num):
                        f_id_ = int('{}{:03d}{:03d}'.format(perfix, s_id, f_id))
                        pc_path = str(Path(data_path) / frame_infos_dict[f_id_]['point_cloud']['velodyne_path'])
                        pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1,8)
                        seg_label = pc[:, 7].astype(np.int32)
                        if seg_label.any():
                            pc_lidar = pc[:, :3]
                            record_pointer = croper.auto_crop_one_frame(pc_lidar, seg_label, record_pointer)
                            _min = max(f_id - 2, 0)
                            _max = f_id + 5
                            crop_pointer_scene[_min:_max] = record_pointer
                            label_ind[_min:_max] = True

                    flag = True
                    for f_id in range(f_num):
                        if flag and label_ind[f_id]:
                            crop_pointer_scene[:f_id] = crop_pointer_scene[f_id]
                            flag = False
                        elif (not flag) and (~label_ind[f_id]):
                            crop_pointer_scene[f_id] = crop_pointer_scene[f_id-1]
                        if (f_id == (f_num-1)) and flag:
                            print(f'Warning: No segmentation label in scene {s_id}.')                
                    for f_id in range(f_num):
                        if label_ind[f_num-f_id-1]:
                            crop_pointer_scene[:f_id] = crop_pointer_scene[f_id]
                            break
                    crop_pointer_dict[s_id] = crop_pointer_scene
            except Exception as err:
                if crop_pointer_dict != {}:
                    croper.save_crop_pointer_dict(save_path, crop_pointer_dict)
                    print('An error has occurred. Now saving {} scenes parameters of croping ground to '.format(
                        len(crop_pointer_dict.keys())) + save_path)
                raise err
            if save_path is not None:
                croper.save_crop_pointer_dict(save_path, crop_pointer_dict)
                print('Save parameters of croping ground to ' + save_path)
        return crop_pointer_dict
    
    def generate_scene_flow(self, 
                            data_path,
                            save_path,
                            frame_idx,
                            frame_infos_dict,
                            remove_outside=False,
                            pc_num_features=8,
                            croper=None,
                            crop_pointer_dict=None,
                            downsample=None,
                            num_workers=8):
        """Create scene flow dataset.

        Args:
            
        """
        def map_func(idx):
            err_count = 0
            err_tole = 100
            try:
                if idx + 1 in frame_infos_dict:
                    s_id = idx // 1000 % 1000
                    f_id = idx % 1000

                    info1 = frame_infos_dict[idx]
                    info2 = frame_infos_dict[idx+1]

                    pc_info1 = info1['point_cloud']
                    pc_info2 = info2['point_cloud']
                    pc_path1 = str(Path(data_path) / pc_info1['velodyne_path'])
                    pc_path2 = str(Path(data_path) / pc_info2['velodyne_path'])
                    pc1 = np.fromfile(
                        pc_path1, dtype=np.float32, count=-1).reshape([-1, pc_num_features])
                    pc2 = np.fromfile(
                        pc_path2, dtype=np.float32, count=-1).reshape([-1, pc_num_features])

                    calib1 = info1['calib']
                    calib2 = info2['calib']
                    rect1 = calib1['R0_rect']
                    rect2 = calib2['R0_rect']
                    Trv2c1 = calib1['Tr_velo_to_cam']
                    Trv2c2 = calib2['Tr_velo_to_cam']
                    P2_1 = calib1['P2']
                    P2_2 = calib2['P2']

                    if remove_outside:
                        image_info1 = info1['image']
                        pc1 = box_np_ops.remove_outside_points(
                            pc1, rect1, Trv2c1, P2_1, image_info1['image_shape'])
                        image_info2 = info2['image']
                        pc2 = box_np_ops.remove_outside_points(
                            pc2, rect2, Trv2c2, P2_2, image_info2['image_shape'])

                    # points_v = points_v[points_v[:, 0] > 0]
                    annos1 = info1['annos']
                    annos2 = info2['annos']
                    num_obj1 = len([n for n in annos1['name'] if n != 'DontCare'])
                    num_obj2 = len([n for n in annos2['name'] if n != 'DontCare'])
                    # annos1 = filter_kitti_anno(annos1, ['DontCare'])
                    # annos2 = filter_kitti_anno(annos2, ['DontCare'])
                    dims1 = annos1['dimensions'][:num_obj1]
                    dims2 = annos2['dimensions'][:num_obj2]
                    loc1 = annos1['location'][:num_obj1]
                    loc2 = annos2['location'][:num_obj2]
                    rots1 = annos1['rotation_y'][:num_obj1]
                    rots2 = annos2['rotation_y'][:num_obj2]
                    gt_boxes_camera1 = np.concatenate([loc1, dims1, rots1[..., np.newaxis]],
                                                    axis=1)
                    gt_boxes_camera2 = np.concatenate([loc2, dims2, rots2[..., np.newaxis]],
                                                    axis=1)
                    gt_boxes_lidar1 = box_np_ops.box_camera_to_lidar(
                        gt_boxes_camera1, rect1, Trv2c1)
                    gt_boxes_lidar2 = box_np_ops.box_camera_to_lidar(
                        gt_boxes_camera2, rect2, Trv2c2)
                    T_boxes2lidar1 = box_np_ops.get_T_box_to_lidar(gt_boxes_lidar1)
                    T_boxes2lidar2 = box_np_ops.get_T_box_to_lidar(gt_boxes_lidar2)
                    T_lidar2boxes1 = np.linalg.inv(T_boxes2lidar1)
                    pc1_xyz = pc1[:, :3]
                    pc2_xyz = pc2[:, :3]
                    pc1_pad = np.concatenate([pc1_xyz,np.ones((pc1.shape[0],1))], axis=-1)
                    fg_ind1 = box_np_ops.points_in_rbbox(pc1_xyz, gt_boxes_lidar1)
                    fg_ind2 = box_np_ops.points_in_rbbox(pc2_xyz, gt_boxes_lidar2)
                    bg_ind1 = ~(fg_ind1.sum(axis=1).astype(np.bool))
                    bg_ind2 = ~(fg_ind2.sum(axis=1).astype(np.bool))
                    bg_pc1_xyz = pc1_xyz[bg_ind1]
                    bg_pc2_xyz = pc2_xyz[bg_ind2]
                    
                    if croper is not None:
                        crop_pointer1 = crop_pointer_dict[s_id][f_id]
                        crop_pointer2 = crop_pointer_dict[s_id][f_id+1]
                        crop_mask1 = croper.compute_crop(bg_pc1_xyz[:,:3], crop_pointer1)
                        crop_mask2 = croper.compute_crop(bg_pc2_xyz[:,:3], crop_pointer2)
                        bg_pc1 = pc1[bg_ind1][crop_mask1]
                        bg_pc2 = pc2[bg_ind2][crop_mask2]
                        bg_pc1_pad = pc1_pad[bg_ind1][crop_mask1]
                    else:
                        bg_pc1 = pc1[bg_ind1]
                        bg_pc2 = pc2[bg_ind2]
                        bg_pc1_pad = pc1_pad[bg_ind1]
                    
                    # compute background flow
                    pose1 = info1['pose']
                    pose2 = info2['pose']
                    bg_flow = (bg_pc1_pad @ pose1.T @ np.linalg.inv(pose2).T - bg_pc1_pad)[:,:3]

                    # compute foreground flow
                    track_id1 = annos1['track_id'][:num_obj1]
                    track_id2 = annos2['track_id'][:num_obj2]
                    fg_pc1, fg_pc2, fg_flow = [], [], []
                    for i in range(len(track_id1)):
                        if track_id1[i] in track_id2:
                            i2 = track_id2.index(track_id1[i])
                            fg_ind1_ = fg_ind1[:, i]
                            fg_ind2_ = fg_ind2[:, i2]
                            fg_pc1.append(pc1[fg_ind1_])
                            fg_pc2.append(pc2[fg_ind2_])
                            pc1_obj = pc1_pad[fg_ind1_]
                            T_lidar2boxes1_obj = T_lidar2boxes1[i]
                            T_boxes2lidar2_obj = T_boxes2lidar2[i2]
                            pc1_obj_in_2 = pc1_obj @ T_lidar2boxes1_obj.T @ T_boxes2lidar2_obj.T
                            flow_obj = (pc1_obj_in_2 - pc1_obj)[:,:3]
                            fg_flow.append(flow_obj)
                    fg_pc1 = np.concatenate(fg_pc1, axis=0) if fg_pc1 != [] else np.empty((0,pc1.shape[-1]))
                    fg_pc2 = np.concatenate(fg_pc2, axis=0) if fg_pc2 != [] else np.empty((0,pc2.shape[-1]))
                    fg_flow = np.concatenate(fg_flow, axis=0) if fg_flow != [] else np.empty((0,3))
                    
                    new_pc1 = np.concatenate([bg_pc1, fg_pc1], axis=0)
                    new_pc2 = np.concatenate([bg_pc2, fg_pc2], axis=0)
                    gt_flow = np.concatenate([bg_flow, fg_flow], axis=0).astype(np.float32)
                    fg_indices1 = np.concatenate([
                        np.zeros(bg_pc1.shape[0], dtype=np.bool),
                        np.ones(fg_pc1.shape[0], dtype=np.bool)], axis=0)
                    fg_indices2 = np.concatenate([
                        np.zeros(bg_pc2.shape[0], dtype=np.bool),
                        np.ones(fg_pc2.shape[0], dtype=np.bool)], axis=0)
                    
                    if (downsample is not None) and (isinstance(downsample, int)):
                        n1 = new_pc1.shape[0]
                        n2 = new_pc2.shape[0]
                        if downsample < 10000:
                            samp1 = random.sample(range(n1), downsample)
                            samp2 = random.sample(range(n2), downsample)
                        else:
                            samp1 = np.random.choice(n1, size=downsample, replace=False)
                            samp2 = np.random.choice(n2, size=downsample, replace=False)
                        new_pc1 = new_pc1[samp1]
                        new_pc2 = new_pc2[samp2]
                        gt_flow = gt_flow[samp1]
                        fg_indices1 = fg_indices1[samp1]
                        fg_indices2 = fg_indices2[samp2]
                    
                    save_dir = Path(save_path) / '{:03d}'.format(s_id) / 'sf_data'
                    save_dir.mkdir(parents=True, exist_ok=True)
                    file_save_path = save_dir / '{:07d}'.format(idx)
                    np.savez(file_save_path, pc1=new_pc1, pc2=new_pc2, gt=gt_flow, fg_index=fg_indices1, fg_index_t=fg_indices2, proj_mat=P2_1)
            except Exception as err:
                print('An error occurred during the generation of {:07d}.npz'.format(idx))
                err_count += 1
                if err_count <= err_tole:
                    print(err)
                else:
                    raise err
            return None
        
        print('Start to generate flow data.')
        with futures.ThreadPoolExecutor(num_workers) as executor:
            list(tqdm(executor.map(map_func, frame_idx), total=len(frame_idx), ncols=100))

    def get_frame_idx(self, data_path, scene_id_list, split):
        imageset_folder = Path(data_path) / 'ImageSets'
        imageset_path = str(imageset_folder / (split.lower() + '.txt'))
        with open(imageset_path, 'r') as f:
            lines = f.readlines()
        frame_idx = []
        for line in lines:
            scene_id = int(line[1:4])
            if scene_id in scene_id_list:
                frame_idx.append(int(line[1:]))
                #frame_idx.append(int(line))
        return frame_idx