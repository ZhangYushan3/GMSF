import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# GMFlow Transformer
def single_head_full_attention(q, k, v):
    # q, k, v: [B, L, C]
    assert q.dim() == k.dim() == v.dim() == 3

    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** .5)  # [B, L, L]
    attn = torch.softmax(scores, dim=2)  # [B, L, L]
    out = torch.matmul(attn, v)  # [B, L, C]

    return out

def multi_head_full_attention(q, k, v, num_head):
    # q, k, v: [B, L, C]
    assert q.dim() == k.dim() == v.dim() == 3
    assert q.size(2) % num_head == 0
    B, N ,C = q.shape

    q, k, v = q.view(B, N, num_head, C//num_head), k.view(B, N, num_head, C//num_head), v.view(B, N, num_head, C//num_head)
    # Transpose for attention dot product: B, num_head, N, C//num_head
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    scores = torch.matmul(q, k.transpose(2, 3)) / (q.size(3) ** .5)  # [B, num_head, N, N]
    attn = torch.softmax(scores, dim=-1)  # [B, num_head, N, N]
    out = torch.matmul(attn, v)  # [B, num_head, N, C//num_head]
    out = out.transpose(1, 2).contiguous().view(B, N, -1)

    return out

class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model=128,
                 nhead=1,
                 no_ffn=False,
                 ffn_dim_expansion=4,
                 ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.no_ffn = no_ffn

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, source, target,
                height=None,
                width=None,
                ):
        # source, target: [B, L, C]
        query, key, value = source, target, target

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C]
        value = self.v_proj(value)  # [B, L, C]

        if self.nhead == 1:
            message = single_head_full_attention(query, key, value)  # [B, L, C]
        else:
            message = multi_head_full_attention(query, key, value, self.nhead)  # [B, L, C]
        message = self.merge(message)  # [B, L, C]
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(self,
                 d_model=128,
                 nhead=1,
                 ffn_dim_expansion=4,
                 ):
        super(TransformerBlock, self).__init__()

        self.self_attn = TransformerLayer(d_model=d_model,
                                          nhead=nhead,
                                          no_ffn=True,
                                          ffn_dim_expansion=ffn_dim_expansion,
                                          )

        self.cross_attn_ffn = TransformerLayer(d_model=d_model,
                                               nhead=nhead,
                                               ffn_dim_expansion=ffn_dim_expansion,
                                               )

    def forward(self, source, target,
                height=None,
                width=None,
                ):
        # source, target: [B, L, C]

        # self attention
        source = self.self_attn(source, source,
                                height=height,
                                width=width,
                                )

        # cross attention and ffn
        source = self.cross_attn_ffn(source, target,
                                     height=height,
                                     width=width,
                                     )

        return source

class FeatureTransformer3D(nn.Module):
    def __init__(self,
                 num_layers=6,
                 d_model=128,
                 nhead=1,
                 ffn_dim_expansion=4,
                 ):
        super(FeatureTransformer3D, self).__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model,
                             nhead=nhead,
                             ffn_dim_expansion=ffn_dim_expansion,
                             )
            for i in range(num_layers)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, feature1,
                **kwargs,
                ):

        b, c, n = feature0.shape
        assert self.d_model == c

        feature0 = feature0.permute(0, 2, 1)  # [B, N, C]
        feature1 = feature1.permute(0, 2, 1)  # [B, N, C]

        # concat feature0 and feature1 in batch dimension to compute in parallel
        concat0 = torch.cat((feature0, feature1), dim=0)  # [2B, N, C]
        concat1 = torch.cat((feature1, feature0), dim=0)  # [2B, N, C]


        for i, layer in enumerate(self.layers):
            concat0 = layer(concat0, concat1,
                            height=n,
                            width=1,
                            )

            # update feature1
            concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)

        feature0, feature1 = concat0.chunk(chunks=2, dim=0)  # [B, H*W, C]

        # reshape back
        feature0 = feature0.view(b, n, c).permute(0, 2, 1).contiguous()  # [B, C, N]
        feature1 = feature1.view(b, n, c).permute(0, 2, 1).contiguous()  # [B, C, N]

        return feature0, feature1

# point transformer
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    #return torch.cdist(src, dst)**2
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    #raw_size = idx.size()
    #idx = idx.reshape(raw_size[0], -1)
    #res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    #return res.reshape(*raw_size, -1)

    B,S,K = idx.shape
    B,N,C = points.shape
    points = points.reshape(B*N,C)
    batch_dim = torch.arange(0, B, device = idx.device)
    idx = (batch_dim[:,None] * N +  idx.reshape(B,S*K)).flatten()
    return points[idx].reshape(B,S,K,C)

class TransformerBlock_PT(nn.Module):
    def __init__(self, d_points, d_model, nhead, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    def forward(self, xyz, features): # xyz: b x 3 x n, features: b x n x f

        xyz = torch.permute(xyz, (0,2,1))
        dists = square_distance(xyz, xyz) # b x n x n
        #knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_idx = torch.topk(dists, self.k, dim = -1, largest = False, sorted = False).indices
        knn_xyz = index_points(xyz, knn_idx) # b x n x k x 3
        
        pre = features # b x n x f
        x = self.fc1(features) # b x n x C
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        # q: b x n x C, k: b x k x C, v: b x k x C
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x C
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc) # b x n x k x C
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x C
        
        #res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc) # b x n x C
        res = (attn * (v+pos_enc)).sum(dim=-2)
        res = self.fc2(res) + pre # b x n x f
        return res, attn

class FeatureTransformer3D_PT(nn.Module):
    def __init__(self,
                 num_layers=1,
                 d_points=16,
                 nhead=1,
                 ffn_dim_expansion=4,
                 ):
        super(FeatureTransformer3D_PT, self).__init__()

        self.d_points = d_points
        self.nhead = nhead
        self.d_model = d_points * ffn_dim_expansion

        self.layers = nn.ModuleList([
            TransformerBlock_PT(d_points=self.d_points,
                               d_model=self.d_model,
                               nhead = self.nhead, 
                               k=16,
                               )
            for i in range(num_layers)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xyz0, xyz1, feature0, feature1,
                **kwargs,
                ):

        b, c, n = feature0.shape
        assert self.d_points == c

        feature0 = feature0.permute(0, 2, 1)  # [B, N, C]
        feature1 = feature1.permute(0, 2, 1)  # [B, N, C]

#        # concat feature0 and feature1 in batch dimension to compute in parallel
#        concat0 = torch.cat((feature0, feature1), dim=0)  # [2B, N, C]
#        concat1 = torch.cat((feature1, feature0), dim=0)  # [2B, N, C]

        for i, layer in enumerate(self.layers):
            feature0, _ = layer(xyz0, feature0)
            feature1, _ = layer(xyz1, feature1)

#            # update feature1
#            concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)

#        feature0, feature1 = concat0.chunk(chunks=2, dim=0)  # [B, H*W, C]

        # reshape back
        feature0 = feature0.view(b, n, c).permute(0, 2, 1).contiguous()  # [B, C, N]
        feature1 = feature1.view(b, n, c).permute(0, 2, 1).contiguous()  # [B, C, N]

        return feature0, feature1