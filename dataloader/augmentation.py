import cv2
import torch
import torchvision
import numpy as np

def flip_point_cloud(pc, image_h, image_w, f, cx, cy, flip_mode):
    assert flip_mode in ['lr', 'ud']
    pc_x, pc_y, depth = pc[..., 0], pc[..., 1], pc[..., 2]

    image_x = cx + (f / depth) * pc_x
    image_y = cy + (f / depth) * pc_y

    if flip_mode == 'lr':
        image_x = image_w - 1 - image_x
    else:
        image_y = image_h - 1 - image_y

    pc_x = (image_x - cx) * depth / f
    pc_y = (image_y - cy) * depth / f
    pc = np.concatenate([pc_x[:, None], pc_y[:, None], depth[:, None]], axis=-1)

    return pc


def flip_scene_flow(pc1, flow_3d, image_h, image_w, f, cx, cy, flip_mode):
    new_pc1 = flip_point_cloud(pc1, image_h, image_w, f, cx, cy, flip_mode)
    new_pc1_warp = flip_point_cloud(pc1 + flow_3d[:, :3], image_h, image_w, f, cx, cy, flip_mode)
    return np.concatenate([new_pc1_warp - new_pc1, flow_3d[:, 3:]], axis=-1)


def random_flip_pc(image_h, image_w, pc1, pc2, flow_3d, f, cx, cy, flip_mode):
    assert flow_3d.shape[1] <= 4
    assert flip_mode in ['lr', 'ud']

    if np.random.rand() < 0.5:  # do nothing
        return pc1, pc2, flow_3d

    # flip point clouds
    new_pc1 = flip_point_cloud(pc1, image_h, image_w, f, cx, cy, flip_mode)
    new_pc2 = flip_point_cloud(pc2, image_h, image_w, f, cx, cy, flip_mode)

    # flip scene flow
    new_flow_3d = flip_scene_flow(pc1, flow_3d, image_h, image_w, f, cx, cy, flip_mode)

    return new_pc1, new_pc2, new_flow_3d


def joint_augmentation_pc(pc1, pc2, flow_3d, f, cx, cy, image_h, image_w):
    # FlyingThings3D
    enabled = True
    random_horizontal_flip = True
    random_vertical_flip = True

    if random_horizontal_flip:
        pc1, pc2, flow_3d = random_flip_pc(
            image_h, image_w, pc1, pc2, flow_3d, f, cx, cy, flip_mode='lr'
        )

    if random_vertical_flip:
        pc1, pc2, flow_3d = random_flip_pc(
            image_h, image_w, pc1, pc2, flow_3d, f, cx, cy, flip_mode='ud'
        )

    return pc1, pc2, flow_3d, f, cx, cy