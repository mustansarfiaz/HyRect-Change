#!/usr/bin/env python3

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model
from .cd_modules import Decoder
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
from typing import Tuple, Union
from functools import partial
import numpy as np


class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2) #(b c h w)
        x = self.conv(x) #(b c h w)
        x = x.permute(0, 2, 3, 1) #(b h w c)
        return x
    

class RelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
        
    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H*W, 2) #(H*W 2)
        mask = grid[:, None, :] - grid[None, :, :] #(H*W H*W 2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]  #(n H*W H*W)
        return mask
    
    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :] #(l l)
        mask = mask.abs() #(l l)
        mask = mask * self.decay[:, None, None]  #(n l l)
        return mask
    
    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:

            retention_rel_pos = self.decay.exp()

        elif chunkwise_recurrent:
            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = (mask_h, mask_w)

        else:
            mask = self.generate_2d_decay(slen[0], slen[1]) #(n l l)
            retention_rel_pos = mask

        return retention_rel_pos
    
class MaSA(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        rel_pos: mask: (n l l)
        '''
        bsz, h, w, _ = x.size()
        mask = rel_pos
        
        assert h*w == mask.size(1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        qr = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1)
        kr = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1)


        qr = qr.flatten(2, 3) #(b n l d1)
        kr = kr.flatten(2, 3) #(b n l d1)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d2)
        vr = vr.flatten(2, 3) #(b n l d2)
        qk_mat = qr @ kr.transpose(-1, -2) #(b n l l)
        qk_mat = qk_mat + mask  #(b n l l)
        qk_mat = torch.softmax(qk_mat, -1) #(b n l l)
        output = torch.matmul(qk_mat, vr) #(b n l d2)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output
    
class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        x = x.permute(0, 2, 3, 1).contiguous() #(b h w c)
        x = self.norm(x) #(b h w c)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x



class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Local_block(nn.Module):
    r""" Local Feature Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_rate=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv = nn.Linear(dim, dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        x = shortcut + self.drop_path(x)
        return x

class DifferenceEncoder(nn.Module):
    def __init__(self, dims=[256, 512, 1024, 2048], kernel_size=3, heads=[4, 8, 16, 32], embedding_dim=256,
                  init_values=[2, 2, 2, 2], heads_ranges=[3, 3, 3, 3], chunkwise_recurrent=False):
        super().__init__()
       
        self.conv11 = Local_block(dims[0]//2)
        self.conv22 = Local_block(dims[1]//2)
        self.conv33 = Local_block(dims[2]//2)
        self.conv44 = Local_block(dims[3]//2)

        self.conv1 = nn.Conv2d(in_channels=dims[0]*2, out_channels=dims[0]//2, kernel_size=1, stride=1, padding=1//2, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dims[1]*2, out_channels=dims[1]//2, kernel_size=1, stride=1, padding=1//2, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dims[2]*2, out_channels=dims[2]//2, kernel_size=1, stride=1, padding=1//2, bias=True)
        self.conv4 = nn.Conv2d(in_channels=dims[3]*2, out_channels=dims[3]//2, kernel_size=1, stride=1, padding=1//2, bias=True)

               
        self.proj1 = nn.Conv2d(in_channels=dims[0], out_channels=dims[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2 = nn.Conv2d(in_channels=dims[1], out_channels=dims[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3 = nn.Conv2d(in_channels=dims[2], out_channels=dims[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4 = nn.Conv2d(in_channels=dims[3], out_channels=dims[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.chunkwise_recurrent = chunkwise_recurrent
        self.Relpos1 = RelPos2d(dims[0]//2, heads[0], init_values[0], heads_ranges[0])
        self.Relpos2 = RelPos2d(dims[1]//2, heads[1], init_values[1], heads_ranges[1])
        self.Relpos3 = RelPos2d(dims[2]//2, heads[2], init_values[2], heads_ranges[2])
        self.Relpos4 = RelPos2d(dims[3]//2, heads[3], init_values[3], heads_ranges[3])

        self.attn1 = MaSA(dims[0]//2,  heads[0])
        self.attn2 = MaSA(dims[1]//2,  heads[1])
        self.attn3 = MaSA(dims[2]//2,  heads[2])
        self.attn4 = MaSA(dims[3]//2,  heads[3])

        self.norm1 = LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        self.norm4 = LayerNorm(dims[0], eps=1e-6, data_format="channels_first")

        self.norm11 = LayerNorm(dims[0]//2, eps=1e-6, data_format="channels_first")
        self.norm22 = LayerNorm(dims[1]//2, eps=1e-6, data_format="channels_first")
        self.norm33 = LayerNorm(dims[2]//2, eps=1e-6, data_format="channels_first")
        self.norm44 = LayerNorm(dims[3]//2, eps=1e-6, data_format="channels_first")

        self.multiscale_fusion = nn.Conv2d(in_channels=dims[0]*4, out_channels=embedding_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.multiscale_norm = nn.BatchNorm2d(embedding_dim)

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, pre, post):
        x1, x2, x3, x4 = pre
        y1, y2, y3, y4 = post

        B, C, H, W = x1.shape
        d1 = torch.cat([x1, y1], dim=1)
        d2 = torch.cat([x2, y2], dim=1)
        d3 = torch.cat([x3, y3], dim=1)
        d4 = torch.cat([x4, y4], dim=1)

        d1 = self.norm11(self.conv1(d1))
        spat1 = self.gelu(self.conv11(d1))
        
        
        d1 = d1.permute(0,2,3,1).contiguous()
        b, h, w, d = d1.size()  #(b h w c)        
        rel_pos1 = self.Relpos1((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        attn1 = self.attn1(d1, rel_pos=rel_pos1, chunkwise_recurrent=self.chunkwise_recurrent, incremental_state=None)
        attn1 = attn1.permute(0, 3, 1, 2).contiguous()  #(b c h w) 
        local2global = torch.sigmoid(attn1)
        global2local = torch.sigmoid(spat1)
        local_feat = spat1 * local2global 
        global_feat = attn1 * global2local
        f1 = self.norm1(self.relu(self.proj1(torch.cat([local_feat, global_feat], dim=1))))

        d2 = self.norm22(self.conv2(d2))
        spat2 = self.gelu(self.conv22(d2))
        
        
        d2 = d2.permute(0,2,3,1).contiguous()
        b, h, w, d = d2.size()  #(b h w c)        
        rel_pos2 = self.Relpos2((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        attn2 = self.attn2(d2, rel_pos=rel_pos2, chunkwise_recurrent=self.chunkwise_recurrent, incremental_state=None)
        attn2 = attn2.permute(0, 3, 1, 2).contiguous()  #(b c h w)
        local2global = torch.sigmoid(attn2)
        global2local = torch.sigmoid(spat2)
        local_feat = spat2 * local2global 
        global_feat = attn2 * global2local       
        f2 = self.norm2(self.relu(self.proj2(torch.cat([local_feat, global_feat], dim=1))))

        d3 = self.norm33(self.conv3(d3))
        spat3 = self.gelu(self.conv33(d3))
       
        
        d3 = d3.permute(0,2,3,1).contiguous()
        b, h, w, d = d3.size()  #(b h w c)        
        rel_pos3 = self.Relpos3((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        attn3 = self.attn3(d3, rel_pos=rel_pos3, chunkwise_recurrent=self.chunkwise_recurrent, incremental_state=None)
        attn3 = attn3.permute(0, 3, 1, 2).contiguous()  #(b c h w) 
        local2global = torch.sigmoid(attn3)
        global2local = torch.sigmoid(spat3)
        local_feat = spat3 * local2global 
        global_feat = attn3 * global2local      
        f3 = self.norm3(self.relu(self.proj3(torch.cat([local_feat, global_feat], dim=1))))

        d4 = self.norm44(self.conv4(d4))
        spat4 = self.gelu(self.conv44(d4))
        
        
        d4 = d4.permute(0,2,3,1).contiguous()
        b, h, w, d = d4.size()  #(b h w c)        
        rel_pos4 = self.Relpos4((h, w), chunkwise_recurrent=self.chunkwise_recurrent)
        attn4 = self.attn4(d4, rel_pos=rel_pos4, chunkwise_recurrent=self.chunkwise_recurrent, incremental_state=None)
        attn4 = attn4.permute(0, 3, 1, 2).contiguous()  #(b c h w) 
        local2global = torch.sigmoid(attn4)
        global2local = torch.sigmoid(spat4)
        local_feat = spat4 * local2global 
        global_feat = attn4 * global2local       
        f4 = self.norm4(self.relu(self.proj4(torch.cat([local_feat, global_feat], dim=1))))

        f2 = F.interpolate(f2, size=(H, W), mode='bilinear')
        f3 = F.interpolate(f3, size=(H, W), mode='bilinear')
        f4 = F.interpolate(f4, size=(H, W), mode='bilinear')

        x = torch.cat((f1,f2,f3,f4), dim=1)
        x = self.multiscale_fusion(x)
        x = self.multiscale_norm(self.relu(x))

        return x


class ChangeBindModel(nn.Module):
    def __init__(self, embed_dim=256, encoder_dims=[256, 512, 1024, 2048], freeze_backbone=False):
        super().__init__()

        self.visual_encoder = create_model('resnet50', pretrained=False, features_only=True)
        self.difference_encoder = DifferenceEncoder(dims=encoder_dims, embedding_dim=embed_dim)
        self.decoder = Decoder(embedding_dim=embed_dim)
        
        if freeze_backbone:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

    def forward_visual_features(self, x):
        _, x1, x2, x3, x4 = self.visual_encoder(x)
        return x1, x2, x3, x4

    def encode_difference_features(self, pre_feats, post_feats):
        x = self.difference_encoder(pre_feats, post_feats)
        return x

    def forward(self, pre_img, post_img):
        # extract visual features
        #pre_img, post_img = images

        x1, x2, x3, x4 = self.forward_visual_features(pre_img)
        y1, y2, y3, y4 = self.forward_visual_features(post_img)

        # extract difference features
        diff_feats = self.encode_difference_features([x1,x2,x3,x4], [y1,y2,y3,y4])

        pred = self.decoder(diff_feats)

        return pred


if __name__ == "__main__":
    model = ChangeBindModel()

    from fvcore.nn import FlopCountAnalysis, flop_count_table
    input_res = (3, 224, 224)
    input = torch.ones(()).new_empty((1, *input_res), dtype=next(model.parameters()).dtype,
                                     device=next(model.parameters()).device)
    #model.eval()
    flops = FlopCountAnalysis(model, (input, input))
    print(flop_count_table(flops))
