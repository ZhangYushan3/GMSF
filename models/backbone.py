import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=32, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1) # (BNk)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :] # (BNk C)
    feature = feature.view(batch_size, num_points, k, num_dims) # (B N k C)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # (B N k C)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous() # (B 2C N k)
  
    return feature


class PointNet(nn.Module):
    def __init__(self, output_channels=128):
        super(PointNet, self).__init__()
        self.outchannel = output_channels
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, self.outchannel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.outchannel)

    def forward(self, x): # x torch.Size([B, 3, 8192])
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x


class DGCNN(nn.Module):
    def __init__(self, output_channels=128, k=32): # 32
        super(DGCNN, self).__init__()
        self.outchannel = output_channels
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm1d(self.outchannel)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(320, self.outchannel, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x): # x torch.Size([B, 3, 8192])
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k) # (B 2C N k)
        x = self.conv1(x) # (B 64 N k)
        x1 = x.max(dim=-1, keepdim=False)[0] # (B 64 N)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        return x

class MLP(nn.Module):
    def __init__(self, output_channels=128):
        super(MLP, self).__init__()
        self.outchannel = output_channels
        self.conv1 = nn.Conv1d(3, self.outchannel, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(self.outchannel, self.outchannel, kernel_size=1, bias=False)
        self.activation_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x): # x torch.Size([B, 3, 8192])
        x = self.activation_fn(self.conv1(x))
        x = self.activation_fn(self.conv2(x))
        return x
