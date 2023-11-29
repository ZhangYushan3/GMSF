import torch
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from dataloader.datasets import build_train_dataset
from utils import misc
from utils.logger import Logger
from loss.sceneflow_loss import sceneflow_loss_func
from evaluate.evaluate import validate_things, validate_kitti, validate_waymo
from models.gmsf import GMSF

def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='checkpoints', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default='things_subset', type=str,
                        help='training stage on different datasets (things_subset / things_subset_non-occluded / things_flownet3d / waymo )')
    parser.add_argument('--val_dataset', default=['things', 'kitti15'], type=str, nargs='+', 
                        help='waymo / things / kitti15')
    # training
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--scheduler', default='OneCycleLR', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--seed', default=326, type=int)
    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_true',
                        help='strict resume while loading pretrained weights')
    # model: learnable parameters
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--backbone', default='DGCNN', type=str, help='feature extraction backbone (DGCNN / pointnet / mlp)')
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_pt_layers', default=1, type=int)
    parser.add_argument('--num_transformer_layers', default=10, type=int)
    # evaluation
    parser.add_argument('--eval', action='store_true',
                        help='evaluation after training done')

    # log
    parser.add_argument('--summary_freq', default=500, type=int, help='Summary frequency to tensorboard (iterations)')
    parser.add_argument('--save_ckpt_freq', default=10000, type=int, help='Save checkpoint frequency (steps)')
    parser.add_argument('--val_freq', default=1000, type=int, help='validation frequency in terms of training steps')
    parser.add_argument('--save_latest_ckpt_freq', default=500, type=int)
    parser.add_argument('--num_steps', default=600000, type=int)

    # loss
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='exponential weighting')
    return parser


def main(args):
    # precheck before training
    print_info = not args.eval
    if print_info:
        print(args)
        misc.save_args(args)
        misc.check_path(args.checkpoint_dir)
        misc.save_command(args.checkpoint_dir)
    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GMSF(backbone=args.backbone,
                 feature_channels=args.feature_channels,
                 ffn_dim_expansion=args.ffn_dim_expansion,
                 num_transformer_pt_layers=args.num_transformer_pt_layers,
                 num_transformer_layers=args.num_transformer_layers).to(device)

    if not args.eval:
        print('Model definition:')
        print(model)
    # multiple GPUs
    if torch.cuda.device_count() > 1:
        print('Use %d GPUs' % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    # count training parameters
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num)
    if not args.eval:
        save_name = '%d_parameters' % num_params
        open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    # training
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    start_epoch = 0
    start_step = 0
    # resume
    if args.resume:
        print('Load checkpoint: %s' % args.resume)
        loc = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)
        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_step = checkpoint['step']
            start_epoch = checkpoint['epoch']
        print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))
    # load training datset
    train_dataset = build_train_dataset(args)
    print('=> {} training samples found in the training set'.format(len(train_dataset)))

    # batch training dataset (26025 * [img1, img2, flow_gt, valid])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True,
                                               sampler=None)

    last_epoch = start_step if args.resume and start_step > 0 else -1
    if args.scheduler == 'OneCycleLR':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, args.lr,
            args.num_steps + 10,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy='cos',
            last_epoch=last_epoch,
        )
    # tensorboard
    summary_writer = SummaryWriter(args.checkpoint_dir)
    logger = Logger(lr_scheduler, summary_writer, args.summary_freq,
                    start_step=start_step)
    # start training
    total_steps = start_step
    epoch = start_epoch
    print('Start training')
    while total_steps < args.num_steps:
        model.train()

        for i, sample in enumerate(train_loader):
            # train one step
            pcs = sample['pcs'] # 6*8192
            flow_3d = sample['flow_3d'].to(device)
            pc1 = pcs[:,:,0:3].to(device)
            pc2 = pcs[:,:,3:6].to(device)

            results_dict_point = model(pc0 = pc1, pc1 = pc2)
            sceneflow_preds = results_dict_point['flow_preds']

            loss, metrics_3d = sceneflow_loss_func(sceneflow_preds, flow_3d)

            if isinstance(loss, float):
                continue
            if torch.isnan(loss):
                continue
            metrics_3d.update({'total_loss': loss.item()})
            # more efficient zero_grad
            for param in model_without_ddp.parameters():
                param.grad = None
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            lr_scheduler.step()
            # Tensorboard / wandb
            logger.push(metrics_3d)
            total_steps += 1
            # save checkpoint of specific epoch
            if total_steps % args.save_ckpt_freq == 0 or total_steps == args.num_steps:
                print('Save checkpoint at step: %d' % total_steps)
                checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': total_steps,
                    'epoch': epoch,
                }, checkpoint_path)
            # always save the latest model for resuming training
            if total_steps % args.save_latest_ckpt_freq == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': total_steps,
                    'epoch': epoch,
                }, checkpoint_path)
  
            if total_steps >= args.num_steps:
                print('Training done')

                return
        epoch += 1

def val(args):
    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model
    model = GMSF(backbone=args.backbone,
                 feature_channels=args.feature_channels,
                 ffn_dim_expansion=args.ffn_dim_expansion,
                 num_transformer_pt_layers=args.num_transformer_pt_layers,
                 num_transformer_layers=args.num_transformer_layers).to(device)
    # Multiple GPUs
    if torch.cuda.device_count() > 1:
        print('Use %d GPUs' % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    # resume checkpoints
    if args.resume:
        #print('Load checkpoint: %s' % args.resume)
        loc = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)
    # evaluate
    if args.eval:
        import timeit
        val_results = {}

        if 'waymo' in args.val_dataset:
            start = timeit.default_timer()
            results_dict = validate_waymo(args.stage,
                                          model_without_ddp,
                                          )
            val_results.update(results_dict)
            stop = timeit.default_timer()
            print('Time: ', stop - start)  

        if 'things' in args.val_dataset:
            start = timeit.default_timer()
            results_dict = validate_things(args.stage,
                                           model_without_ddp,
                                           )
            val_results.update(results_dict)
            stop = timeit.default_timer()
            print('Time: ', stop - start)  

        if 'kitti15' in args.val_dataset:
            start = timeit.default_timer()
            results_dict = validate_kitti(args.stage,
                                          model_without_ddp,
                                          )
            val_results.update(results_dict)
            stop = timeit.default_timer()
            print('Time: ', stop - start)  


        return

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.eval:
        val(args)
    else:
        main(args)