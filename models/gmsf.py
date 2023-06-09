import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import FeatureTransformer3D, FeatureTransformer3D_PT
from .matching import global_correlation_softmax_3d, SelfCorrelationSoftmax3D
from .backbone import PointNet, DGCNN, MLP

def parallel2perspect(xyz, perspect_camera_info, parallel_camera_info):
    src_x, src_y, src_z = xyz[:, 0, :], xyz[:, 1, :], xyz[:, 2, :]  # [batch_size, n_points]

    # scaling
    perspect_h, perspect_w = perspect_camera_info['sensor_h'], perspect_camera_info['sensor_w']
    parallel_h, parallel_w = parallel_camera_info['sensor_h'], parallel_camera_info['sensor_w']

    scale_ratio_w = (parallel_w - 1) / (perspect_w - 1)
    scale_ratio_h = (parallel_h - 1) / (perspect_h - 1)

    src_x = (src_x + (parallel_w - 1) / 2) / scale_ratio_w
    src_y = (src_y + (parallel_h - 1) / 2) / scale_ratio_h
    src_z = src_z / min(scale_ratio_w, scale_ratio_h)
    # transformation
    batch_size, n_points = src_x.shape
    f = perspect_camera_info['f'][:, None].expand([batch_size, n_points])
    cx = perspect_camera_info['cx'][:, None].expand([batch_size, n_points])
    cy = perspect_camera_info['cy'][:, None].expand([batch_size, n_points])

    dst_z = torch.exp((src_z - 1) / f)
    dst_x = (src_x - cx) * dst_z / f
    dst_y = (src_y - cy) * dst_z / f

    return torch.cat([
        dst_x[:, None, :],
        dst_y[:, None, :],
        dst_z[:, None, :],
    ], dim=1)

def perspect2parallel(xyz, perspect_camera_info, parallel_camera_info):
    src_x, src_y, src_z = xyz[:, 0, :], xyz[:, 1, :], xyz[:, 2, :]  # [batch_size, n_points]

    # transformation
    batch_size, n_points = src_x.shape
    f = perspect_camera_info['f'][:, None].expand([batch_size, n_points])
    cx = perspect_camera_info['cx'][:, None].expand([batch_size, n_points])
    cy = perspect_camera_info['cy'][:, None].expand([batch_size, n_points])

    dst_x = cx + (f / src_z) * src_x
    dst_y = cy + (f / src_z) * src_y
    dst_z = f * torch.log(src_z) + 1

    # scaling
    perspect_h, perspect_w = perspect_camera_info['sensor_h'], perspect_camera_info['sensor_w']
    parallel_h, parallel_w = parallel_camera_info['sensor_h'], parallel_camera_info['sensor_w']

    scale_ratio_w = (parallel_w - 1) / (perspect_w - 1)
    scale_ratio_h = (parallel_h - 1) / (perspect_h - 1)

    dst_xyz = torch.cat([
        dst_x[:, None, :] * scale_ratio_w - (parallel_w - 1) / 2,
        dst_y[:, None, :] * scale_ratio_h - (parallel_h - 1) / 2,
        dst_z[:, None, :] * min(scale_ratio_w, scale_ratio_h),
    ], dim=1)

    return dst_xyz

class GMSF(nn.Module):
    def __init__(self,
                 backbone=None,
                 feature_channels=128,
                 ffn_dim_expansion=4,
                 num_transformer_pt_layers=1,
                 num_transformer_layers=8,
                 ):
        super(GMSF, self).__init__()

        self.backbone = backbone
        self.feature_channels = feature_channels

        # PointNet
        if self.backbone=='pointnet':
            self.pointnet = PointNet(output_channels = self.feature_channels)
            feature_channels = self.feature_channels
        #MLP
        if self.backbone=='mlp':
            self.mlp = MLP(output_channels = self.feature_channels)
            feature_channels = self.feature_channels
        # DGCNN
        if self.backbone=='DGCNN':
            self.DGCNN = DGCNN(output_channels = self.feature_channels)
            feature_channels = self.feature_channels

        self.num_transformer_layers = num_transformer_layers
        self.num_transformer_pt_layers = num_transformer_pt_layers

        # Transformer
        if self.num_transformer_layers > 0:
            self.transformer = FeatureTransformer3D(num_layers=num_transformer_layers,
                                                d_model=feature_channels,
                                                ffn_dim_expansion=ffn_dim_expansion,
                                                )
        if self.num_transformer_pt_layers > 0:
            self.transformer_PT = FeatureTransformer3D_PT(num_layers=num_transformer_pt_layers,
                                                        d_points=feature_channels,
                                                        ffn_dim_expansion=ffn_dim_expansion,
                                                        )

        # self correlation with self-feature similarity
        self.feature_flow_attn = SelfCorrelationSoftmax3D(in_channels=feature_channels)


    def forward(self, pc0, pc1, origin_h, origin_w,
                intrinsics=None,
                **kwargs,
                ):

        results_dict = {}
        flow_preds = []

        persp_cam_info = {
            'projection_mode': 'perspective',
            'sensor_h': origin_h,
            'sensor_w': origin_w,
            'f': intrinsics[:, 0],
            'cx': intrinsics[:, 1],
            'cy': intrinsics[:, 2],
        }
        parallel_sensor_size = (
            origin_h // 32,
            origin_w // 32,
        ) 
        paral_cam_info = {
            'projection_mode': 'parallel',
            'sensor_h': parallel_sensor_size[0],
            'sensor_w': parallel_sensor_size[1],
            'cx': (parallel_sensor_size[1] - 1) / 2,
            'cy': (parallel_sensor_size[0] - 1) / 2,
        } 

        pc0 = torch.permute(pc0, (0, 2, 1)) # torch.Size([batch, n_point, 3]) -> torch.Size([batch, 3, n_point])
        pc1 = torch.permute(pc1, (0, 2, 1)) # torch.Size([batch, n_point, 3]) -> torch.Size([batch, 3, n_point])

        # # Inverse Depth Sampling
        pc0 = perspect2parallel(pc0, persp_cam_info, paral_cam_info)
        pc1 = perspect2parallel(pc1, persp_cam_info, paral_cam_info)


        xyzs1, xyzs2 = pc0, pc1
        if self.backbone=='pointnet':
            feats0_3d = self.pointnet(xyzs1)
            feats1_3d = self.pointnet(xyzs2)
        if self.backbone=='mlp':
            feats0_3d = self.mlp(xyzs1)
            feats1_3d = self.mlp(xyzs2)
        if self.backbone=='DGCNN':
            feats0_3d = self.DGCNN(xyzs1)
            feats1_3d = self.DGCNN(xyzs2)

        sceneflow = None

        feature0, feature1 = feats0_3d, feats1_3d
        # Transformer
        if self.num_transformer_pt_layers > 0:
            feature0, feature1 = self.transformer_PT(xyzs1, xyzs2, 
                                                    feature0, feature1,
                                                    )

        if self.num_transformer_layers > 0:
            feature0, feature1 = self.transformer(feature0, feature1,
                                                )
        # global matching
        # flow prediction with cros-attn
        flow_pred = global_correlation_softmax_3d(feature0, feature1, xyzs1, xyzs2)[0]
        sceneflow = sceneflow + flow_pred if sceneflow is not None else flow_pred
        if self.training:
            flow_preds.append(sceneflow)
        # flow refinement with self-attn
        sceneflow = self.feature_flow_attn(feature0, sceneflow.detach())
        flow_preds.append(sceneflow)

        # Inverse Depth Sampling
        for idx, flow12_3d in enumerate(flow_preds):
            flow_preds[idx] = parallel2perspect(xyzs1 + flow12_3d, persp_cam_info, paral_cam_info) - \
                            parallel2perspect(xyzs1, persp_cam_info, paral_cam_info)

        results_dict.update({'flow_preds': flow_preds})

        return results_dict
