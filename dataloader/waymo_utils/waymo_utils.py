import numpy as np
from collections import OrderedDict
from concurrent import futures as futures
from pathlib import Path
from skimage import io
from tqdm import tqdm


def get_kitti_info_path(idx,
                        prefix,
                        info_type,
                        file_tail,
                        relative_path,
                        exist_check):
    filename = '{:07d}'.format(idx) + file_tail
    scene_path = '{:03d}'.format(idx // 1000 % 1000)
    prefix = Path(prefix)
    file_path = Path(scene_path) / info_type / filename
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(idx,
                   prefix,
                   relative_path=True,
                   exist_check=True,
                   info_type='image_2'):
    return get_kitti_info_path(idx, prefix, info_type, '.png',
                               relative_path, exist_check)


def get_label_path(idx,
                   prefix,
                   relative_path=True,
                   exist_check=True,
                   info_type='label_2'):
    return get_kitti_info_path(idx, prefix, info_type, '.txt',
                               relative_path, exist_check)


def get_velodyne_path(idx,
                      prefix,
                      relative_path=True,
                      exist_check=True):
    return get_kitti_info_path(idx, prefix, 'velodyne', '.bin',
                               relative_path, exist_check)


def get_calib_path(idx,
                   prefix,
                   relative_path=True,
                   exist_check=True):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt',
                               relative_path, exist_check)


def get_pose_path(idx,
                  prefix,
                  relative_path=True,
                  exist_check=True):
    return get_kitti_info_path(idx, prefix, 'pose', '.txt',
                               relative_path, exist_check)


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'track_id': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    annotations['track_id'] = [x[16] for x in content]
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_waymo_frame_info(path,
                         label_info=True,
                         velodyne=True,
                         pc_num_features=6,
                         calib=True,
                         pose=True,
                         image=False,
                         with_imageshape=False,
                         extend_matrix=True,
                         frame_idx=[],
                         num_workers=8,
                         relative_path=True):
    """
    Get info of all frames:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            frame_idx: ...
            frame_path: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)

    def map_func(idx):
        info = {'idx': idx}
        pc_info = {'num_features': pc_num_features}
        calib_info = {}

        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx, path, relative_path)
            points = np.fromfile(
                Path(path) / pc_info['velodyne_path'], dtype=np.float32)
            points = np.copy(points).reshape(-1, pc_info['num_features'])
        info['point_cloud'] = pc_info
        if label_info:
            label_path = get_label_path(
                idx,
                path,
                relative_path,
                info_type='label_all')
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        if image:
            image_info = {}
            image_info['image_path'] = get_image_path(
                idx,
                path,
                relative_path,
                info_type='image_0')
            if with_imageshape:
                img_path = image_info['image_path']
                if relative_path:
                    img_path = str(root_path / img_path)
                image_info['image_shape'] = np.array(
                    io.imread(img_path).shape[:2], dtype=np.int32)
            info['image'] = image_info
        if calib:
            calib_path = get_calib_path(
                idx, path, relative_path=False)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            P4 = np.array([float(info) for info in lines[4].split(' ')[1:13]
                           ]).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
                P4 = _extend_matrix(P4)
            R0_rect = np.array([
                float(info) for info in lines[5].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['P4'] = P4
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            info['calib'] = calib_info
        if pose:
            pose_path = get_pose_path(
                idx, path, relative_path=False)
            info['pose'] = np.loadtxt(pose_path)

        if annotations is not None:
            info['annos'] = annotations
            info['annos']['camera_id'] = info['annos'].pop('score')
            add_difficulty_to_annos(info)
        return info

    print('Start to load infos.')
    with futures.ThreadPoolExecutor(num_workers) as executor:
        frame_infos = list(tqdm(executor.map(map_func, frame_idx), total=len(frame_idx), ncols=100))
        print(frame_infos)
    
    frame_infos_dict = {}
    for info in frame_infos:
        frame_infos_dict[info['idx']] = info
    return frame_infos_dict


def kitti_anno_to_label_file(annos, folder):
    folder = Path(folder)
    for anno in annos:
        image_idx = anno['metadata']['image_idx']
        label_lines = []
        for j in range(anno['bbox'].shape[0]):
            label_dict = {
                'name': anno['name'][j],
                'alpha': anno['alpha'][j],
                'bbox': anno['bbox'][j],
                'location': anno['location'][j],
                'dimensions': anno['dimensions'][j],
                'rotation_y': anno['rotation_y'][j],
                'score': anno['score'][j],
            }
            label_line = kitti_result_line(label_dict)
            label_lines.append(label_line)
        label_file = folder / '{:07d}.txt'.format(image_idx)
        label_str = '\n'.join(label_lines)
        with open(label_file, 'w') as f:
            f.write(label_str)


def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool)
    hard_mask = np.ones((len(dims), ), dtype=np.bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff


def kitti_result_line(result_dict, precision=4):
    prec_float = '{' + ':.{}f'.format(precision) + '}'
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError('you must specify a value for {}'.format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError('unknown key. supported key:{}'.format(
                res_dict.keys()))
    return ' '.join(res_line)