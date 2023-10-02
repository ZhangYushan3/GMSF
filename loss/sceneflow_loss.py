import torch

def sceneflow_loss_func(flow_preds, flow_gt, 
                   gamma=0.9,
                   **kwargs,
                   ):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        level_target = flow_gt
        if level_target.shape[1] == 4:
            flow_mask = level_target[:, 3, :] > 0
            diff = flow_preds[i] - level_target[:, :3, :]
            epe_l1 = torch.pow(diff.abs().sum(dim=1) + 0.01, 0.4)[flow_mask].mean()
        else:
            diff = flow_preds[i] - level_target
            epe_l1 = torch.pow(diff.abs().sum(dim=1) + 0.01, 0.4).mean()
        
        i_weight = gamma ** (n_predictions - i - 1)
        flow_loss += i_weight * epe_l1

        # compute endpoint error
        target_3d = flow_gt[:,:3,:]
        if flow_gt.shape[1] == 4:
            target_3d_mask = (flow_gt[:,3,:] > 0) * 1.
        else:
            target_3d_mask = torch.ones_like(flow_gt[:,0,:])
        #diff = torch.zeros_like(flow_preds[-1]) - target_3d
        diff = flow_preds[-1] - target_3d
        epe3d_map = torch.linalg.norm(diff, dim=1) * target_3d_mask
        epe3d_bat = epe3d_map.sum(dim=1) / target_3d_mask.sum(dim=1)

        # compute 5cm accuracy
        acc5_3d_map = (epe3d_map < 0.05).float() * target_3d_mask
        acc5_3d_bat = acc5_3d_map.sum(dim=1) / target_3d_mask.sum(dim=1)

        # compute 5cm accuracy
        acc10_3d_map = (epe3d_map < 0.10).float() * target_3d_mask
        acc10_3d_bat = acc10_3d_map.sum(dim=1) / target_3d_mask.sum(dim=1)

    metrics = {
        'epe3d': epe3d_bat.mean().item(),
        'acc3d_5cm':acc5_3d_bat.mean().item(),
        'acc3d_10cm':acc10_3d_bat.mean().item(),
    }

    return flow_loss, metrics
