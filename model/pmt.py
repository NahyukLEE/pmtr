
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange, repeat
from scipy.linalg import orth

from common.node import index_select

class PMTBlock(nn.Module):
    def __init__(self, c_channels, f_channels, nheads, att_type, radius, neighbor=False):
        super(PMTBlock, self).__init__()

        self.att_type = att_type
        self.neighbor = neighbor
        c_inch = c_channels[0]
        f_inch = f_channels[0]

        self.pmt, self.norms, self.att_layer1, self.att_layer2 = [], [], [], []
        for c_ch, f_ch, nhead in zip(c_channels[1:], f_channels[1:], nheads):
            c_outch, f_outch = c_ch, f_ch

            if neighbor:
                self.pmt.append(PMTNeighbor(c_inch, f_inch, c_outch, f_outch, nhead))
                
            else:
                self.pmt.append(PMT(c_inch, f_inch, c_outch, f_outch, nhead))

                att_layer1 = nn.Sequential(
                        nn.Linear(1 if att_type == 'dist' else 3, nhead),
                        nn.ReLU(inplace=True),
                        nn.Linear(nhead, nhead)
                        )
                self.att_layer1.append(att_layer1)

                att_layer2 = nn.Sequential(
                        nn.Linear(1 if att_type == 'dist' else 3, nhead),
                        nn.ReLU(inplace=True),
                        nn.Linear(nhead, nhead)
                        )
                self.att_layer2.append(att_layer2)

            self.norms.append(PairGroupNorm(c_outch, c_outch))
            c_inch, f_inch = c_outch, f_outch
        
        self.pmt, self.norms = nn.ModuleList(self.pmt), nn.ModuleList(self.norms)

        if not neighbor:
            self.att_layer1, self.att_layer2 = nn.ModuleList(self.att_layer1), nn.ModuleList(self.att_layer2)
        
        self.lrelu = PairActivation('lrelu')
        self.tanh = nn.Tanh()
        self.out = nn.Linear(c_channels[-1] * f_channels[-1], c_channels[-1] * f_channels[-1])
        self.radius = radius

    def _positional_embedding(self, position):
        dist = torch.cdist(position, position)
        dist_mask = (dist > self.radius)
        if self.att_type == 'dist':
            pos_emb = dist.unsqueeze(-1)
        elif self.att_type == 'disp':
            pos_emb = position.unsqueeze(0) - position.unsqueeze(1)
        pos_emb = pos_emb.unsqueeze(0)

        return pos_emb, dist_mask

    def forward(self, src_pos, trg_pos, src_feat, trg_feat):
        src_pos_emb, src_dist_mask = self._positional_embedding(src_pos.squeeze(0))
        trg_pos_emb, trg_dist_mask = self._positional_embedding(trg_pos.squeeze(0))

        c_inch, f_inch = self.pmt[0].c_inch, self.pmt[0].f_inch 
        src_feat = rearrange(src_feat, '() s (f c) -> s f c', c=c_inch, f=f_inch)
        trg_feat = rearrange(trg_feat, '() s (f c) -> s f c', c=c_inch, f=f_inch)
        srctrg = tuple((F.normalize(src_feat, dim=-1), F.normalize(trg_feat, dim=-1)))

        for pmt, norm, att_layer1, att_layer2 in zip(self.pmt, self.norms, self.att_layer1, self.att_layer2):
            src_att, trg_att = att_layer1(src_pos_emb).squeeze(0), att_layer2(trg_pos_emb).squeeze(0)
            srctrg = pmt(srctrg, src_att, trg_att, src_dist_mask, trg_dist_mask)
            srctrg = norm(srctrg)
            srctrg = self.lrelu(srctrg)

        src, trg = srctrg
        src = self.out(rearrange(src, 's f c -> s (f c)')).unsqueeze(0)
        trg = self.out(rearrange(trg, 's f c -> s (f c)')).unsqueeze(0)

        return src, trg

    def forward_with_neighbor(self, feat, point, neighbor, length):
        src_len, trg_len = length
        if neighbor.max() >= point.size(0):
            nei_mask = neighbor == point.size(0)
            nei_mask = nei_mask[:src_len], nei_mask[src_len:]
            nei_pts = torch.cat([point, torch.zeros_like(point[:1, :])], dim=0)[neighbor]
        else:
            nei_pts = point[neighbor]
            nei_mask = None, None
        srctrg_pts = point[:src_len], point[src_len:]
        src_feat, trg_feat = feat[:src_len], feat[src_len:]
        srctrg_nei = nei_pts[:src_len], nei_pts[src_len:]

        c_inch, f_inch = self.pmt[0].c_inch, self.pmt[0].f_inch
        src_feat = rearrange(src_feat, 's (f c) -> s f c', c=c_inch, f=f_inch)
        trg_feat = rearrange(trg_feat, 's (f c) -> s f c', c=c_inch, f=f_inch)
        srctrg = tuple((F.normalize(src_feat, dim=-1), F.normalize(trg_feat, dim=-1)))

        for pmt, norm in zip(self.pmt, self.norms):
            srctrg = pmt(srctrg, srctrg_pts, srctrg_nei, neighbor, length, nei_mask)
            srctrg = norm(srctrg)
            srctrg = self.lrelu(srctrg)
        out = rearrange(torch.cat(srctrg, dim=0), 's f c -> s (f c)')
        
        return out

class PMT(nn.Module):
    def __init__(self, c_inch, f_inch, c_outch, f_outch, nhead):
        super(PMT, self).__init__()
        self.nhead = nhead
        self.c_inch, self.c_outch, self.f_inch, self.f_outch = c_inch, c_outch, f_inch, f_outch

        self.w_src = nn.Linear(self.c_inch, self.nhead * self.c_outch)
        self.w_trg = nn.Linear(self.c_inch, self.nhead * self.c_outch)
        self.out_src = nn.Linear(self.c_outch * self.nhead, self.c_outch)
        self.out_trg = nn.Linear(self.c_outch * self.nhead, self.c_outch)

        # Proxy Tensor
        proxy = nn.Parameter(rearrange(nn.Linear(f_inch, f_outch * nhead, bias=False).weight, '(h o) i -> h i o', h=nhead, i=f_inch, o=f_outch))
        self.proxy = nn.Parameter(proxy)

    def forward(self, srctrg, src_att, trg_att, src_mask, trg_mask):
        # 1. Initialization
        src, trg = srctrg
        proxy = self.proxy

        # 2. Linear projection on 4D channel
        src, trg = self.w_src(src), self.w_trg(trg)
        src = rearrange(src, 's f (c h) -> s f c h', c=self.c_outch, h=self.nhead)
        trg = rearrange(trg, 's f (c h) -> s f c h', c=self.c_outch, h=self.nhead)

        # 3. ProxyTensor
        src = torch.einsum('s f c h, h f o -> s o c h', src, proxy)
        trg = torch.einsum('s f c h, h f o -> s o c h', trg, proxy)

        # 4. Apply attention
        src_att[src_mask.unsqueeze(-1).repeat(1, 1, src_att.size(-1))] = float('-inf')
        src_att = F.softmax(src_att, dim=-2)
        src = torch.einsum('s j h, j f c h -> s f c h', src_att, src)

        trg_att[trg_mask.unsqueeze(-1).repeat(1, 1, trg_att.size(-1))] = float('-inf')
        trg_att = F.softmax(trg_att, dim=-2)
        trg = torch.einsum('s j h, j f c h -> s f c h', trg_att, trg)

        # 5. Aggregate heads
        src = self.out_src(rearrange(src, 's f c h -> s f (c h)'))
        trg = self.out_trg(rearrange(trg, 's f c h -> s f (c h)'))
        
        return (src, trg)

class PMTNeighbor(nn.Module):
    def __init__(self, c_inch, f_inch, c_outch, f_outch, nhead):
        super(PMTNeighbor, self).__init__()
        self.nhead = nhead
        self.c_inch, self.c_outch, self.f_inch, self.f_outch = c_inch, c_outch, f_inch, f_outch

        self.w_src = nn.Linear(self.c_inch, self.nhead * self.c_outch)
        self.w_trg = nn.Linear(self.c_inch, self.nhead * self.c_outch)
        self.out_src = nn.Linear(self.c_outch * self.nhead, self.c_outch)
        self.out_trg = nn.Linear(self.c_outch * self.nhead, self.c_outch)

        self.src_att = nn.Sequential(*[nn.Linear(1, nhead), nn.ReLU(inplace=True), nn.Linear(nhead, nhead)])
        self.trg_att = nn.Sequential(*[nn.Linear(1, nhead), nn.ReLU(inplace=True), nn.Linear(nhead, nhead)])

        # Proxy Tensor
        proxy = nn.Parameter(rearrange(nn.Linear(f_inch, f_outch * nhead, bias=False).weight, '(h o) i -> h i o', h=nhead, i=f_inch, o=f_outch))
        self.proxy = nn.Parameter(proxy)

    def forward(self, srctrg, srctrg_pts, srctrg_nei, neighbor, length, nei_mask):
        # 1. Initialization
        src_len, trg_len = length
        src_pts, trg_pts = srctrg_pts
        src, trg = srctrg
        src_nei_pts, trg_nei_pts = srctrg_nei
        src_nei_mask, trg_nei_mask = nei_mask

        proxy = self.proxy

        # 2. Linear projection on 4D channel
        src, trg = self.w_src(src), self.w_trg(trg)
        src = rearrange(src, 's f (c h) -> s f c h', c=self.c_outch, h=self.nhead)
        trg = rearrange(trg, 's f (c h) -> s f c h', c=self.c_outch, h=self.nhead)

        # 3. ProxyTensor
        src = torch.einsum('s f c h, h f o -> s o c h', src, proxy)
        trg = torch.einsum('s f c h, h f o -> s o c h', trg, proxy)

        # 4. Compute attention
        src_att = self.src_att((src_pts.unsqueeze(1) - src_nei_pts).pow(2).sum(dim=-1, keepdim=True))
        trg_att = self.trg_att((trg_pts.unsqueeze(1) - trg_nei_pts).pow(2).sum(dim=-1, keepdim=True))
        if src_nei_mask is not None and trg_nei_mask is not None:
            src_att[src_nei_mask.unsqueeze(-1).repeat(1, 1, self.nhead)] = float('-inf')
            trg_att[trg_nei_mask.unsqueeze(-1).repeat(1, 1, self.nhead)] = float('-inf')
        src_att = F.softmax(src_att, dim=1)
        trg_att = F.softmax(trg_att, dim=1)

        # 5. Apply attention
        srctrg_nei_feat = torch.cat([torch.cat([src, trg], dim=0), torch.zeros_like(src[:1])], dim=0)[neighbor]
        src_nei_feat, trg_nei_feat = srctrg_nei_feat[:src_len], srctrg_nei_feat[src_len:]
        src = torch.einsum('s a h, s a f c h -> s f c h', src_att, src_nei_feat)
        trg = torch.einsum('s a h, s a f c h -> s f c h', trg_att, trg_nei_feat)

        # 5. Aggregate heads
        src = self.out_src(rearrange(src, 's f c h -> s f (c h)'))
        trg = self.out_trg(rearrange(trg, 's f c h -> s f (c h)'))

        return (src, trg)

class PairGroupNorm(nn.Module):
    def __init__(self, group, outch):
        super(PairGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(group, outch)

    def forward(self, srctrg):
        src, trg = srctrg
        src = rearrange(self.norm(rearrange(src, 's f c -> s c f')), 's c f -> s f c')
        trg = rearrange(self.norm(rearrange(trg, 's f c -> s c f')), 's c f -> s f c')

        return (src, trg)

class PairActivation(nn.Module):
    def __init__(self, activation_type):
        super(PairActivation, self).__init__()
        activations = {'relu': nn.ReLU(inplace=True),
                       'sigmoid': nn.Sigmoid(),
                       'lrelu': nn.LeakyReLU(),
                       }
        self.activation = activations[activation_type]

    def forward(self, srctrg):
        src, trg = srctrg
        src = self.activation(src)
        trg = self.activation(trg)

        return (src, trg)


