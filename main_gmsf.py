import torch
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
#import wandb
import torch.nn.functional as F

from dataloader.datasets import build_train_dataset
from utils import misc
from utils.logger import Logger
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed
from loss.sceneflow_loss import sceneflow_loss_func
from evaluate.evaluate import validate_things, validate_kitti
from models.gmsf import GMSF

def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='checkpoints', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default='things_subset', type=str,
                        help='training stage on different datasets (things_subset / things_subset_non-occluded / things_flownet3d )')
    parser.add_argument('--val_dataset', default=['things','kitti15'], type=str, nargs='+')
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
    parser.add_argument('--feature_channels_point', default=128, type=int)
    parser.add_argument('--backbone', default='DGCNN', type=str, help='feature extraction backbone (DGCNN / pointnet / mlp)')
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_pt_layers', default=1, type=int)
    parser.add_argument('--num_transformer_layers', default=8, type=int)
    # evaluation
    parser.add_argument('--eval', action='store_true',
                        help='evaluation after training done')
    parser.add_argument('--save_eval_to_file', action='store_true')

    # log
    parser.add_argument('--summary_freq', default=500, type=int, help='Summary frequency to tensorboard (iterations)')
    parser.add_argument('--save_ckpt_freq', default=10000, type=int, help='Save checkpoint frequency (steps)')
    parser.add_argument('--val_freq', default=1000, type=int, help='validation frequency in terms of training steps')
    parser.add_argument('--save_latest_ckpt_freq', default=500, type=int)
    parser.add_argument('--num_steps', default=600000, type=int)

    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    # loss
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='exponential weighting')
    return parser


def main(args):
    # precheck before training
    print_info = not args.eval
    if print_info and args.local_rank == 0:
        print(args)
        misc.save_args(args)
        misc.check_path(args.checkpoint_dir)
        misc.save_command(args.checkpoint_dir)
    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    # distributed training
    if args.launcher == 'none':
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.distributed = True

        # adjust batch size for each gpu
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()

        dist_params = dict(backend='nccl')
        init_dist(args.launcher, **dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        args.gpu_ids = range(world_size)
        device = torch.device('cuda:{}'.format(args.local_rank))
        setup_for_distributed(args.local_rank == 0)

    model_point = GMSF(backbone=args.backbone,
                     feature_channels=args.feature_channels_point,
                     num_head=args.num_head,
                     ffn_dim_expansion=args.ffn_dim_expansion,
                     num_transformer_pt_layers=args.num_transformer_pt_layers,
                     num_transformer_layers=args.num_transformer_layers).to(device)

    if not args.eval:
        print('Model definition:')
        print(model_point)
    # distributed training
    if args.distributed:
        model_point = torch.nn.parallel.DistributedDataParallel(
            model_point.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        model_point_without_ddp = model_point.module
    else:
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            model_point = torch.nn.DataParallel(model_point)
            model_point_without_ddp = model_point.module
        else:
            model_point_without_ddp = model_point
    # count training parameters
    num = sum(p.numel() for p in model_point.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model_point.parameters())
    print('Number of params:', num)
    if not args.eval:
        save_name = '%d_parameters' % num_params
        open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    # training
    optimizer = torch.optim.AdamW(model_point_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    start_epoch = 0
    start_step = 0
    # resume
    if args.resume:
        print('Load checkpoint: %s' % args.resume)
        loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)
        model_point_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)
        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_step = checkpoint['step']
            start_epoch = checkpoint['epoch']
        print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))
    # load training datset
    train_dataset = build_train_dataset(args)
    print('=> {} training samples found in the training set'.format(len(train_dataset)))
    # Multi-processing (train_sampler/shuffle)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank)
    else:
        train_sampler = None
    shuffle = False if args.distributed else True
    # batch training dataset (26025 * [img1, img2, flow_gt, valid])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=shuffle, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True,
                                               sampler=train_sampler)

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
    if args.scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=400000,
            gamma=0.2
        )
    # tensorboard
    if args.local_rank == 0:
        summary_writer = SummaryWriter(args.checkpoint_dir)
        logger = Logger(lr_scheduler, summary_writer, args.summary_freq,
                        start_step=start_step)
#        wandb.init(
#            # Set the project where this run will be logged
#            project="sceneflow", 
#            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
#            name=f"experiment_{args.checkpoint_dir}", 
#            # Track hyperparameters and run metadata
#            config={
#            "architecture": "GMSceneflow",
#            "dataset": "FlyingThings",
#            })
    # start training
    total_steps = start_step
    epoch = start_epoch
    print('Start training')
    while total_steps < args.num_steps:
        model_point.train()
        # mannual change random seed for shuffling every epoch
        if args.distributed:
            train_sampler.set_epoch(epoch)

        for i, sample in enumerate(train_loader):
            # train one step
            pcs = sample['pcs'] # 6*8192
            flow_3d = sample['flow_3d'].to(device)
            pc1 = pcs[:,:,0:3].to(device)
            pc2 = pcs[:,:,3:6].to(device)
            #sceneflow_gt = flow_3d[:,:3,:].to(device)
            #scenevalid = sample['occ_mask_3d'].to(device)
            intrinsics = sample['intrinsics'].to(device)
            input_h = sample['input_h'][0].item()
            input_w = sample['input_w'][0].item()

            results_dict_point = model_point(pc0 = pc1, pc1 = pc2, origin_h=input_h, origin_w=input_w, 
                                 intrinsics = intrinsics
                                 )
            sceneflow_preds = results_dict_point['flow_preds']

            loss, metrics_3d = sceneflow_loss_func(sceneflow_preds, flow_3d)
            if isinstance(loss, float):
                continue
            if torch.isnan(loss):
                continue
            metrics_3d.update({'total_loss': loss.item()})
            # more efficient zero_grad
            for param in model_point_without_ddp.parameters():
                param.grad = None
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model_point.parameters(), args.grad_clip)
            optimizer.step()
            lr_scheduler.step()
            if args.local_rank == 0:
                logger.push(metrics_3d)
                #logger.add_image_summary(img1, img2, flow_preds, flow_gt)
#                if total_steps % args.summary_freq == 0:
#                    wandb.log({"epe3d": metrics_3d['epe3d'], "acc3d_5cm": metrics_3d['acc3d_5cm'], "acc3d_10cm": metrics_3d['acc3d_10cm']})
            total_steps += 1
            # save checkpoint of specific epoch
            if total_steps % args.save_ckpt_freq == 0 or total_steps == args.num_steps:
                if args.local_rank == 0:
                    print('Save checkpoint at step: %d' % total_steps)
                    checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)
                    torch.save({
                        'model': model_point_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': total_steps,
                        'epoch': epoch,
                    }, checkpoint_path)
            # always save the latest model for resuming training
            if total_steps % args.save_latest_ckpt_freq == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')
                if args.local_rank == 0:
                    torch.save({
                        'model': model_point_without_ddp.state_dict(),
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
    # distributed training
    if args.launcher == 'none':
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.distributed = True
        # adjust batch size for each gpu
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()
        dist_params = dict(backend='nccl')
        init_dist(args.launcher, **dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        args.gpu_ids = range(world_size)
        device = torch.device('cuda:{}'.format(args.local_rank))
        setup_for_distributed(args.local_rank == 0)
    # model
    model_point = GMSF(backbone=args.backbone,
                     feature_channels=args.feature_channels_point,
                     num_head=args.num_head,
                     ffn_dim_expansion=args.ffn_dim_expansion,
                     num_transformer_pt_layers=args.num_transformer_pt_layers,
                     num_transformer_layers=args.num_transformer_layers).to(device)
    if args.distributed:
        model_point = torch.nn.parallel.DistributedDataParallel(
            model_point.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        model_point_without_ddp = model_point.module
    else:
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            model_point = torch.nn.DataParallel(model_point)
            model_point_without_ddp = model_point.module
        else:
            model_point_without_ddp = model_point
    # resume checkpoints
    if args.resume:
        #print('Load checkpoint: %s' % args.resume)
        loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)
        model_point_without_ddp.load_state_dict(checkpoint['model'], strict=args.strict_resume)
    # evaluate
    if args.eval:
        val_results = {}
        if 'things' in args.val_dataset:
            results_dict = validate_things(args.stage,
                                           model_point_without_ddp,
                                           )
            val_results.update(results_dict)
        if 'kitti15' in args.val_dataset:
            results_dict = validate_kitti(args.stage,
                                          model_point_without_ddp,
                                          )
            val_results.update(results_dict)

        if args.save_eval_to_file:
            misc.check_path(args.checkpoint_dir)
            # save validation results
            val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
            with open(val_file, 'a') as f:
                f.write('\neval results after training done\n\n')
                metrics = ['chairs_epe', 'chairs_s0_10', 'chairs_s10_40', 'chairs_s40+',
                           'things_clean_epe', 'things_clean_s0_10', 'things_clean_s10_40', 'things_clean_s40+',
                           'things_final_epe', 'things_final_s0_10', 'things_final_s10_40', 'things_final_s40+',
                           'sintel_clean_epe', 'sintel_clean_s0_10', 'sintel_clean_s10_40', 'sintel_clean_s40+',
                           'sintel_final_epe', 'sintel_final_s0_10', 'sintel_final_s10_40', 'sintel_final_s40+',
                           'kitti_epe', 'kitti_f1', 'kitti_s0_10', 'kitti_s10_40', 'kitti_s40+',
                           ]
                eval_metrics = []
                for metric in metrics:
                    if metric in val_results.keys():
                        eval_metrics.append(metric)

                metrics_values = [val_results[metric] for metric in eval_metrics]

                num_metrics = len(eval_metrics)

                # save as markdown format
                f.write(("| {:>20} " * num_metrics + '\n').format(*eval_metrics))
                f.write(("| {:20.3f} " * num_metrics).format(*metrics_values))

                f.write('\n\n')

        return

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.eval:
        val(args)
    else:
        main(args)