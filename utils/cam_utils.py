import re
import cv2
import sys
import logging
import numpy as np
import torch.utils.data
import torch.distributed as dist
from matplotlib.colors import hsv_to_rgb


def init_logging(filename=None, debug=False):
    logging.root = logging.RootLogger('DEBUG' if debug else 'INFO')
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def load_fpm(filename):
    with open(filename, 'rb') as f:
        header = f.readline().rstrip()
        if header.decode("ascii") == 'PF':
            color = True
        elif header.decode("ascii") == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().decode("ascii").rstrip())
        if scale < 0:  # little-endian
            endian = '<'
        else:
            endian = '>'  # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)

    return data


def load_flow(filepath):
    with open(filepath, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Invalid .flo file: incorrect magic number'
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        flow = np.fromfile(f, np.float32, count=2 * w * h).reshape([h, w, 2])

    return flow

def load_flow_png(filepath, scale=64.0):
    # for KITTI which uses 16bit PNG images
    # see 'https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py'
    # The -1 is here to specify not to change the image depth (16bit), and is compatible
    # with both OpenCV2 and OpenCV3
    flow_img = cv2.imread(filepath, -1)
    flow = flow_img[:, :, 2:0:-1].astype(np.float32)
    mask = flow_img[:, :, 0] > 0
    flow = flow - 32768.0
    flow = flow / scale
    return flow, mask

def save_flow_png(filepath, flow, mask=None, scale=64.0):
    assert flow.shape[2] == 2
    assert np.abs(flow).max() < 32767.0 / scale
    flow = flow * scale
    flow = flow + 32768.0

    if mask is None:
        mask = np.ones_like(flow)[..., 0]
    else:
        mask = np.float32(mask > 0)

    flow_img = np.concatenate([
        mask[..., None],
        flow[..., 1:2],
        flow[..., 0:1]
    ], axis=-1).astype(np.uint16)

    cv2.imwrite(filepath, flow_img)

def load_calib(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('P_rect_02'):
                proj_mat = line.split()[1:]
                proj_mat = [float(param) for param in proj_mat]
                proj_mat = np.array(proj_mat, dtype=np.float32).reshape(3, 4)
                assert proj_mat[0, 1] == proj_mat[1, 0] == 0
                assert proj_mat[2, 0] == proj_mat[2, 1] == 0
                assert proj_mat[0, 0] == proj_mat[1, 1]
                assert proj_mat[2, 2] == 1

    return proj_mat



def disp2pc(disp, baseline, f, cx, cy, flow=None):
    h, w = disp.shape
    depth = baseline * f / (disp + 1e-5)

    xx = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    if flow is None:
        x = (xx - cx) * depth / f
        y = (yy - cy) * depth / f
    else:
        x = (xx - cx + flow[..., 0]) * depth / f
        y = (yy - cy + flow[..., 1]) * depth / f

    pc = np.concatenate([
        x[:, :, None],
        y[:, :, None],
        depth[:, :, None],
    ], axis=-1)

    return pc