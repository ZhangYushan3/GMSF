import sys, os
sys.path.append(os.getcwd())

import argparse
from dataloader.waymo_sf_dataset import WaymoSFDataset as WaymoDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create scene flow data')
    parser.add_argument('--dataset_type', type=str, default='waymo')
    args = parser.parse_args()
    if args.dataset_type == 'waymo':
        scene_id_list = list(range(798)) # can be any scene
        dataset = WaymoDataset(data_root='datasets/waymo-open/train_extract')
        dataset.creat_data(
            data_path='datasets/waymo-open/train_extract',
            save_path='datasets/waymo-open/train_extract', 
            scene_id_list=scene_id_list, 
            split='train', # train/valid
            pc_num_features=8, # 8--waymo>=1.3, 6--waymo==1.2
            rm_ground=True,
            crop_params_save_path='./crop_params.npz',
            crop_params_load_path=None, # None means not using existing params
            num_workers=8)