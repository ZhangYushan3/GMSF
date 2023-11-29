import torch
import torch.nn as nn
import torch.nn.functional as F

def global_correlation_softmax_3d(feature0, feature1, xyzs1, xyzs2,
                               ):
    # global correlation
    b, c, n = feature0.shape
    feature0 = feature0.permute(0, 2, 1)  # [B, N, C]
    feature1 = feature1.view(b, c, -1)  # [B, C, N]

    correlation = torch.matmul(feature0, feature1).view(b, n, n) / (c ** 0.5)  # [B, N, N]

    # flow from softmax
    init_grid_1 = xyzs1.to(correlation.device) # [B, 3, N]
    init_grid_2 = xyzs2.to(correlation.device) # [B, 3, N]
    grid_2 = init_grid_2.permute(0, 2, 1)  # [B, N, 3]

    correlation = correlation.view(b, n, n)  # [B, N, N]

    prob = F.softmax(correlation, dim=-1)  # [B, N, N]

    correspondence = torch.matmul(prob, grid_2).view(b, n, 3).permute(0, 2, 1)  # [B, 3, N]

    # when predicting bidirectional flow, flow is the concatenation of forward flow and backward flow
    flow = correspondence - init_grid_1

    return flow, prob


def global_selfcorrelation_softmax_3d(feature0, flow
                               ):
    # self correlation
    b, c, n = feature0.shape
    feature1 = feature0.view(b, c, -1)  # [B, C, N]
    feature0 = feature0.permute(0, 2, 1)  # [B, N, C]

    correlation = torch.matmul(feature0, feature1).view(b, n, n) / (c ** 0.5)  # [B, N, N]

    correlation = correlation.view(b, n, n)  # [B, N, N]

    prob = F.softmax(correlation, dim=-1)  # [B, N, N]

    flow = torch.matmul(prob, flow.permute(0, 2, 1)).view(b, n, 3).permute(0, 2, 1)  # [B, 3, N]

    return flow, prob

class SelfCorrelationSoftmax3D(nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, in_channels,
                 **kwargs,
                 ):
        super(SelfCorrelationSoftmax3D, self).__init__()

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, flow,
                **kwargs,
                ):
        # q, k: feature [B, C, N], v: flow [B, 3, N]

        b, c, n = feature0.size()

        query = feature0.permute(0, 2, 1)  # [B, N, C]

        query = self.q_proj(query)  # [B, N, C]
        key = self.k_proj(query)  # [B, N, C]

        value = flow.view(b, flow.size(1), n).permute(0, 2, 1)  # [B, N, 3]

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, N, N]
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, value)  # [B, N, 3]
        out = out.view(b, n, value.size(-1)).permute(0, 2, 1)  # [B, 3, N]

        return out
