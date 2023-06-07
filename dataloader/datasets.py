from dataloader.kitti import KITTI_flownet3d
from dataloader.flyingthings3d import FlyingThings3D_subset, FlyingThings3D_flownet3d

def build_train_dataset(args):

    if args.stage == 'things_subset':
        train_dataset = FlyingThings3D_subset(split='train', occ=True)
    if args.stage == 'things_subset_non-occluded':
        train_dataset = FlyingThings3D_subset(split='train', occ=False)
    if args.stage == 'things_flownet3d':
        train_dataset = FlyingThings3D_flownet3d(train=True)

#    else:
#        raise ValueError(f'stage {args.stage} is not supported')

    return train_dataset
