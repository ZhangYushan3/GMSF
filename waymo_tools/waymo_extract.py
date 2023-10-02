import os
from waymo_converter import Waymo2KITTI
from tqdm import tqdm

load_dir = 'datasets/waymo-open/scene_flow/train' # raw data train/valid
save_dir = 'datasets/waymo-open/train_extract' # train_extract/valid_extract
scene_num = len(os.listdir(load_dir))

if __name__ == '__main__':
    waymo = Waymo2KITTI(load_dir, save_dir, str(0), scene_num=scene_num)
    print('Total scene :', scene_num)
    print('Strat convert waymo please wait...')
    for i in tqdm(range(scene_num), total=scene_num, ncols=scene_num):
        waymo.convert_one(i)
    print('DONE!')