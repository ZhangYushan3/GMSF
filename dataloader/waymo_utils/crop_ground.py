import numpy as np


class CropGroundAuto(object):
    def __init__(self, 
                 front_left=['x','y'], # Kitti
                 pitch_limit=[-10,10],
                 pitch_reso=2,
                 roll_limit=[-5,5],
                 roll_reso=0.5,
                 z_limit=[-1.5,1.5],
                 z_reso=0.1,
                 dynamic_eval_thresh=True,
                 eval_thresh={'recall':0.85, 'precision':0.9},
                 eval_weight={'recall':0.45, 'precision':0.55}): # waymo seg
        '''Crop ground by searching a plane that best matches the segmentation label. Since
        the segmentation label exists in the interval frame, the plane will be broadcast 
        to the without the segmentation label.

        Args:
            front_left: Axis corresponding to the front and left direction.
                        x, y, z, -x, -y, -z.
            pitch_limit (List[float]): Limit of pitch. 
            pitch_reso (float): Pitch angle resolution.
            roll_limit (List[float]): Limit of pitch.
            roll_reso (float): Roll angle resolution.
            z_limit (List[float]): Limit of vertical direction.
            z_reso (float): vertical resolution.
            dynamic_eval_thresh (bool): Use adaptive eval thresh.
            eval_thresh (dict): Threshold when plane search stops.
            eval_weight (dict): Weight of different eval parameters.
        '''
        self.pitch = np.arange(pitch_limit[0], pitch_limit[1]+0.01, pitch_reso, dtype=np.float32) / 180 * np.pi # n_p
        self.roll = np.arange(roll_limit[0], roll_limit[1]+0.01, roll_reso, dtype=np.float32) / 180 * np.pi # n_r
        rot_mat = self.gen_crop_params(front_left) # (n_p,n_r,3,3)
        self.rot_mat_z = rot_mat[:,:,2,:] # (n_p,n_r,3)
        self.z = np.arange(z_limit[0], z_limit[1]+0.01, z_reso, dtype=np.float32)
        l_p, l_r, l_z = len(self.pitch), len(self.roll), len(self.z)
        self.prz_ind = np.stack(np.meshgrid(
            np.arange(l_p), np.arange(l_r), np.arange(l_z),
            indexing='ij'), axis=-1).reshape(-1,3) # (lp*lr*lz,3)
        
        self.front_left = front_left

        self.dynamic_eval_thresh = dynamic_eval_thresh
        self.eval_thresh = eval_thresh
        self.reset_eval_thresh()
        self.eval_weight = eval_weight

    def gen_crop_params(self, front_left):
        _pitch_mat = np.stack([np.cos(self.pitch),-np.sin(self.pitch),
                               np.sin(self.pitch), np.cos(self.pitch)], axis=1) # (n_p,4)
        pitch_mat = np.expand_dims(np.eye(3), 0).repeat(len(self.pitch), axis=0) # (n_p,3,3)
        _map = {'x':0, 'y':1, 'z':2, '-x':0, '-y':1, '-z':2}
        _ = [0,1,2]
        _.remove(_map[front_left[0]])
        pitch_mat[:,[_[0],_[0],_[1],_[1]],[_[0],_[1],_[0],_[1]]] = _pitch_mat

        _roll_mat = np.stack([np.cos(self.roll),-np.sin(self.roll),
                              np.sin(self.roll), np.cos(self.roll)], axis=1) # (n_r,4)
        roll_mat = np.expand_dims(np.eye(3), 0).repeat(len(self.roll), axis=0) # (n_r,3,3)
        _map = {'x':0, 'y':1, 'z':2, '-x':0, '-y':1, '-z':2}
        _ = [0,1,2]
        _.remove(_map[front_left[1]])
        roll_mat[:,[_[0],_[0],_[1],_[1]],[_[0],_[1],_[0],_[1]]] = _roll_mat

        rot_mat = np.expand_dims(roll_mat, axis=0) @ np.expand_dims(pitch_mat, axis=1) # (n_p,n_r,3,3)
        return rot_mat
    
    def auto_crop_one_frame(self, pc, seg_label, pointer):
        '''
        Waymo segmentation label:
            TYPE_UNDEFINED = 0; 
            TYPE_CAR = 1; 
            TYPE_TRUCK = 2; 
            TYPE_BUS = 3; 
            // Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction 
            // vehicles, RV, limo, tram). 
            TYPE_OTHER_VEHICLE = 4; 
            TYPE_MOTORCYCLIST = 5; 
            TYPE_BICYCLIST = 6; 
            TYPE_PEDESTRIAN = 7; 
            TYPE_SIGN = 8; 
            TYPE_TRAFFIC_LIGHT = 9; 
            // Lamp post, traffic sign pole etc. 
            TYPE_POLE = 10; 
            // Construction cone/pole. 
            TYPE_CONSTRUCTION_CONE = 11; 
            TYPE_BICYCLE = 12; 
            TYPE_MOTORCYCLE = 13; 
            TYPE_BUILDING = 14; 
            // Bushes, tree branches, tall grasses, flowers etc. 
            TYPE_VEGETATION = 15; 
            TYPE_TREE_TRUNK = 16; 
            // Curb on the edge of roads. This does not include road boundaries if 
            // there’s no curb. 
            TYPE_CURB = 17; 
            // Surface a vehicle could drive on. This include the driveway connecting 
            // parking lot and road over a section of sidewalk. 
            TYPE_ROAD = 18; 
            // Marking on the road that’s specifically for defining lanes such as 
            // single/double white/yellow lines. 
            TYPE_LANE_MARKER = 19; 
            // Marking on the road other than lane markers, bumps, cateyes, railtracks 
            // etc. 
            TYPE_OTHER_GROUND = 20; 
            // Most horizontal surface that’s not drivable, e.g. grassy hill, 
            // pedestrian walkway stairs etc. 
            TYPE_WALKABLE = 21; 
            // Nicely paved walkable surface when pedestrians most likely to walk on. 
            TYPE_SIDEWALK = 22; 

        Args:
            pc: (n,3)
            label: (n,)
            pointer: (3,)
        Return:
            record_pointer: (3,)
        '''
        dis_valid = (pc[:, 0] < 50.) & (pc[:, 1] < 50.)
        pc_v, seg_label_v = pc[dis_valid], seg_label[dis_valid]
        valid = (seg_label_v < 14) | (seg_label_v == 18) # | (seg_label == 21)
        pc_v, seg_label_v = pc_v[valid], seg_label_v[valid]
        label = ((seg_label_v > 0) & (seg_label_v < 14))
        if label.sum() < 100:
            pc_v, seg_label_v = pc[dis_valid], seg_label[dis_valid]
            valid = (seg_label_v < 17) | (seg_label_v == 18)
            pc_v, seg_label_v = pc_v[valid], seg_label_v[valid]
            label = ((seg_label_v > 0) & (seg_label_v < 17))
            if label.sum() < 100:
                pc_v, seg_label_v = pc[dis_valid], seg_label[dis_valid]
                valid = (seg_label_v < 17) | (seg_label_v == 18)
                pc_v, seg_label_v = pc_v[valid], seg_label_v[valid]
                label = (seg_label_v < 17)
        
        pos = pc_v[label]
        nag = pc_v[~label]
        nag_ind = np.random.randint(0,nag.shape[0],pos.shape[0])
        nag = nag[nag_ind]
        pc = np.concatenate((pos,nag))
        label = np.concatenate((np.ones(pos.shape[0], dtype=np.bool), np.zeros(nag.shape[0], dtype=np.bool)))

        order = self.gen_order(pointer)
        max_recall = 0.
        max_precision = 0.
        max_scale = 0.
        r_th, p_th, r_w, p_w = \
            self.cur_r_th, self.cur_p_th, \
            self.eval_weight['recall'], self.eval_weight['precision']
        record_pointer = None
        for i in range(order.shape[0]):
            pointer = order[i]
            crop_mask = self.compute_crop(pc, pointer)
            recall, precision = self.eval_crop(crop_mask, label, mode='PR')
            # iou = self.eval_crop(crop_mask, label, mode='IOU')
            if (recall > r_th) and (precision > p_th):
                record_pointer = pointer
                max_recall = recall
                max_precision = precision
                break
            else:
                score = recall * r_w + precision * p_w
                # score = recall * precision
                # score = iou
                if score > max_scale:
                    max_scale = score
                    record_pointer = pointer
                    max_recall = recall
                    max_precision = precision
        if self.dynamic_eval_thresh:
            self.update_eval_thresh(max_recall, max_precision)
        return record_pointer
    
    def gen_order(self, pointer):
        '''Due to the similarity of adjacent frames, it is preferred to search the neighborhood.
        '''
        dist = self.prz_ind - pointer # (lp*lr*lz,3)
        dist = np.max(dist, axis=-1).reshape(-1) # (lp*lr*lz,)
        order_ind = np.argsort(dist)
        order = self.prz_ind[order_ind, :] # (lp*lr*lz,3)
        return order
    
    def compute_crop(self, pc, pointer):
        rot_z = self.rot_mat_z[pointer[0], pointer[1]] # (3,)
        z = self.z[pointer[2]]
        new_pc_z = pc @ rot_z # (n,)
        crop_mask = new_pc_z > z
        return crop_mask
    
    def eval_crop(self, crop_mask, label, mode='PR'):
        '''
        Args:
            crop_mask: (n,) (bool) True means not crop.
            label: (n,) (bool)
            mode: 'PR' or 'IOU'
        '''
        n = crop_mask.shape[0]

        inter = (crop_mask & label).sum()
        if mode == 'PR':
            recall = inter / (np.sum(label, dtype=np.int32) + 0.0001)
            precision = inter / (np.sum(crop_mask, dtype=np.int32) + 0.0001)
            return recall, precision
        elif mode == 'IOU':
            union = (crop_mask | label).sum()
            iou = inter / union
            return iou
        else:
            raise NotImplementedError
    
    def gen_crop_pointer_scene(self, frame_num):
        self.crop_pointer_scene = np.array([[len(self.pitch)//2, len(self.roll)//2, len(self.z)//2]]).repeat(frame_num, axis=0)
        return self.crop_pointer_scene
    
    def gen_crop_pointer_dict(self, frame_num_list, scene_id_list):
        if scene_id_list is None:
            scene_id_list = [i for i in range(len(frame_num_list))]
        elif isinstance(scene_id_list[0], str):
            scene_id_list = [int(i) for i in scene_id_list]
        crop_pointer_list = [
            np.array([[len(self.pitch)//2, len(self.roll)//2, len(self.z)//2]]).repeat(num, axis=0)
            for num in frame_num_list
        ]
        self.crop_pointer_dict = dict(zip(scene_id_list, crop_pointer_list))
        return self.crop_pointer_dict
    
    def save_crop_pointer_dict(self, save_path, crop_pointer_dict=None):
        if crop_pointer_dict is None:
            crop_pointer_dict = self.crop_pointer_dict
        data = {}
        for k, v in crop_pointer_dict.items():
            data['{:03d}'.format(k)] = v
        np.savez(save_path, pitch=self.pitch, roll=self.roll, z=self.z, **data)
    
    def load_crop_pointer_dict(self, load_path):
        data = dict(np.load(load_path))
        self.pitch = data.pop('pitch')
        self.roll = data.pop('roll')
        self.z = data.pop('z')
        rot_mat = self.gen_crop_params(self.front_left) # (n_p,n_r,3,3)
        self.rot_mat_z = rot_mat[:,:,2,:] # (n_p,n_r,3)
        l_p, l_r, l_z = len(self.pitch), len(self.roll), len(self.z)
        self.prz_ind = np.stack(np.meshgrid(
            np.arange(l_p), np.arange(l_r), np.arange(l_z),
            indexing='ij'), axis=-1).reshape(-1,3) # (lp*lr*lz,3)
        
        crop_pointer_dict = {}
        for k, v in data.items():
            crop_pointer_dict[int(k)] = v
        self.crop_pointer_dict = crop_pointer_dict
        return self.crop_pointer_dict

    def reset_eval_thresh(self):
        self.cur_r_th = self.eval_thresh['recall']
        self.cur_p_th = self.eval_thresh['precision']
    
    def update_eval_thresh(self, max_r, max_p):
        u_r_th = (self.cur_r_th + max_r) * 0.5
        if u_r_th > self.eval_thresh['recall']:
            self.cur_r_th = self.eval_thresh['recall']
        elif u_r_th < (self.eval_thresh['recall'] - 0.20):
            self.cur_r_th = self.eval_thresh['recall'] - 0.20
        else:
            self.cur_r_th = u_r_th

        u_p_th = (self.cur_p_th + max_p) * 0.5
        if u_p_th > self.eval_thresh['precision']:
            self.cur_p_th = self.eval_thresh['precision']
        elif u_p_th < (self.eval_thresh['precision'] - 0.10):
            self.cur_p_th = self.eval_thresh['precision'] - 0.10
        else:
            self.cur_p_th = u_p_th


if __name__ == '__main__':
    croper = CropGroundAuto()
    pointer = np.array([0,0,0])
    
    data = np.load('',dtype=np.float32).reshape(-1,8)
    pc = data[:,:3]
    seg_label = data[:,7].astype(np.int32)
    print(np.unique(seg_label))

    crop_mask, recode_pointer = croper.auto_crop_one_frame(pc, seg_label, pointer)
    gt_mask = seg_label < 17
    r, p = croper.eval_crop(crop_mask, gt_mask, mode='PR')
    print('recall:', r, ' precision:', p)

