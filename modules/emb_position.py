import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from functools import lru_cache
import math
from math import pi
from typing import Optional, Any, Union, Tuple
from timm.layers import ndgrid

from .utils import NoInitLinear

class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads,rpb_global):
        super().__init__()
        self.window_size = window_size
        self.rpb_global = rpb_global
        self.num_relative_distance = int((2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    @torch.compiler.disable()
    def forward(self, keep_indices=None, num_prefix_tokens=1,**kwargs):
        if keep_indices is not None:  # B, keep_indices
            B = keep_indices.size(0)
            keep_indices_with_cls = [torch.cat([torch.zeros(1, dtype=torch.long, device=idx.device), idx + 1]) for idx in keep_indices]
            
            new_position_indices = torch.stack([
                self.relative_position_index[idx[:, None], idx[None, :]] 
                for idx in keep_indices_with_cls
            ])
            
            relative_position_bias = self.relative_position_bias_table[new_position_indices.view(-1)].view(
                B,
                new_position_indices.shape[-1],
                new_position_indices.shape[-1], -1)

            if num_prefix_tokens > 1:
                m_tokens = num_prefix_tokens - 1
                pad_size = (0, 0,  # num_heads dimension
                        0, m_tokens,  # last dim (columns)
                        0, m_tokens,  # second last dim (rows)
                        0, 0)  # batch dimension
                relative_position_bias = F.pad(relative_position_bias, pad_size)
                
                # Insert after cls_token (position 1)
                relative_position_bias[:, 1:1+m_tokens] = 0
                relative_position_bias[:, :, 1:1+m_tokens] = 0
                
            return relative_position_bias.permute(0, 3, 1, 2).contiguous()
        
        # Default path
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1, -1)
            
        if num_prefix_tokens > 1:
            m_tokens = num_prefix_tokens - 1
            pad_size = (0, 0,  # num_heads dimension 
                    0, m_tokens,  # columns
                    0, m_tokens)  # rows
            relative_position_bias = F.pad(relative_position_bias, pad_size)
            
            # Insert after cls_token
            relative_position_bias[1:1+m_tokens] = 0
            relative_position_bias[:, 1:1+m_tokens] = 0
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww+1, Wh*Ww+1
    
class ContinuousPositionBias(nn.Module):
    def __init__(self, window_size, num_heads,rpb_global,rpb_dim):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.num_heads = num_heads
        self.rpb_global = rpb_global

        # mlp to generate continuous relative position bias
        # self.cpb_mlp = nn.Sequential(
        #     nn.Linear(2, 512, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, num_heads, bias=False)
        # )

        self.cpb_mlp = nn.Sequential(
            NoInitLinear(2, rpb_dim, bias=True),
            nn.ReLU(inplace=True),
            NoInitLinear(rpb_dim, num_heads, bias=False)
        )

        # Generate coordinate table for regular tokens
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0]).to(torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1]).to(torch.float32)
        relative_coords_table = torch.stack(ndgrid(relative_coords_h, relative_coords_w))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous()

        # Normalize coordinates
        relative_coords_table[..., 0] /= (self.window_size[0] - 1)
        relative_coords_table[..., 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / math.log2(8)

        # Reshape table to match num_relative_distance
        token_coords = relative_coords_table.view(-1, 2)  
        
        self.register_buffer("relative_coords_table", token_coords)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  
        relative_coords[:, :, 0] += window_size[0] - 1  
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww  
      
        self.register_buffer("relative_position_index", relative_position_index)

    @torch.compiler.disable()
    def forward(self, glob_pos=None, keep_indices=None, num_prefix_tokens=1):
        if glob_pos is not None and self.rpb_global:
            glob_pos = glob_pos.squeeze(0)
            bs = glob_pos.size(0)
            
            # 1. 优化位置提取和归一化计算
            if glob_pos.size(-1) == 4:
                pos_all, pos = glob_pos[..., :2], glob_pos[..., 2:]
            else:
                pos_all = glob_pos[0]
                pos = glob_pos[1:, :, [1, 0]]  # 直接指定维度顺序
                pos_all = pos_all[..., [1, 0]]
            
            # 2. 原地归一化计算
            pos = pos / pos_all  # 直接除法，不需要pos_norm变量
            pos.mul_(8)
            pos = torch.sign(pos) * torch.log2(torch.abs(pos) + 1.0) / math.log2(8)
            pos.mul_(2).sub_(1)
            
            # 3. 优化repeat操作
            pos = pos.unsqueeze(1).expand(-1, len(self.relative_coords_table), -1)
            _relative_coords_table = self.relative_coords_table.unsqueeze(0) + pos

            # 4. 直接计算并reshape结果
            relative_position_bias = self.cpb_mlp(_relative_coords_table).view(bs, -1, self.num_heads)

            if keep_indices is not None:  # B, keep_indices
                B = keep_indices.size(0)
          
                # 为每个batch构建新的relative_position_index
                new_position_indices = torch.stack([
                    self.relative_position_index[idx[:, None], idx[None, :]] 
                    for idx in keep_indices
                ])  # B, new_len, new_len                
            else:
                new_position_indices = self.relative_position_index

            # 5. 优化padding操作
            result = torch.zeros(
                (bs, new_position_indices.shape[-1] + num_prefix_tokens,
                 new_position_indices.shape[-1] + num_prefix_tokens, self.num_heads),
                dtype=relative_position_bias.dtype,
                device=relative_position_bias.device
            )

            if keep_indices is not None:
                # 6. 一次性填充数据
                for i in range(bs):
                    result[i, num_prefix_tokens:, num_prefix_tokens:] = relative_position_bias[i, new_position_indices[i].view(-1)].view(
                        new_position_indices.shape[-1], new_position_indices.shape[-1], -1)
                # result[:, 1:, 1:] = relative_position_bias[new_position_indices.view(-1)].view(
                #     bs, new_position_indices.shape[-1],
                #     new_position_indices.shape[-1], -1)
            else:
                # 6. 一次性填充数据
                result[:, num_prefix_tokens:, num_prefix_tokens:] = relative_position_bias[:, new_position_indices.view(-1)].view(
                    bs, new_position_indices.shape[-1],
                    new_position_indices.shape[-1], -1)

            return result.permute(0, 3, 1, 2).contiguous()
        
        else:
            if keep_indices is not None:  # B, keep_indices
                B = keep_indices.size(0)
        
                # 为每个batch构建新的relative_position_index
                new_position_indices = torch.stack([
                    self.relative_position_index[idx[:, None], idx[None, :]] 
                    for idx in keep_indices
                ])  # B, new_len, new_len
      
                _relative_coords_table = self.relative_coords_table
                relative_position_bias_table = self.cpb_mlp(_relative_coords_table).view(-1, self.num_heads)
                # padding以对齐cls_token的shape
                padded_relative_position_bias = torch.zeros((
                B,new_position_indices.shape[-1] + num_prefix_tokens, 
                new_position_indices.shape[-1] + num_prefix_tokens, 
                self.num_heads), dtype=relative_position_bias_table.dtype, device=relative_position_bias_table.device)
                
                # 将计算得到的bias填入padded_relative_position_bias的相应位置
                padded_relative_position_bias[:, num_prefix_tokens:, num_prefix_tokens:] = relative_position_bias_table[new_position_indices.view(-1)].view(
                    B,
                    new_position_indices.shape[-1],
                    new_position_indices.shape[-1], -1
                )

                # 转置维度以匹配注意力的形状
                return padded_relative_position_bias.permute(0, 3, 1, 2).contiguous()  # B, num_heads, new_len, new_len
            else:
                _relative_coords_table = self.relative_coords_table
                relative_position_bias_table = self.cpb_mlp(_relative_coords_table).view(-1, self.num_heads)
                # padding以对齐cls_token的shape
                padded_relative_position_bias = torch.zeros((self.window_size[0] * self.window_size[1] + num_prefix_tokens, 
                self.window_size[0] * self.window_size[1] + num_prefix_tokens, 
                self.num_heads), dtype=relative_position_bias_table.dtype, device=relative_position_bias_table.device)
                
                # 将计算得到的bias填入padded_relative_position_bias的相应位置
                padded_relative_position_bias[num_prefix_tokens:, num_prefix_tokens:] = relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1],
                    self.window_size[0] * self.window_size[1], -1)  

                return padded_relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        
class PPEG(nn.Module):
    def __init__(self, dim=512,k=7,conv_1d=False,bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (5,1), 1, (5//2,0), groups=dim,bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (3,1), 1, (3//2,0), groups=dim,bias=bias)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        
        add_length = H * W - N
        # if add_length >0:
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) 

        if H < 7:
            H,W = 7,7
            zero_pad = H * W - (N+add_length)
            x = torch.cat([x, torch.zeros((B,zero_pad,C),device=x.device,dtype=x.dtype)],dim = 1)
            add_length += zero_pad

        # H, W = int(N**0.5),int(N**0.5)
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # feat_token = x
        cnn_feat = x.transpose(1, 2).view(B, C, H, W).contiguous()
        #cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # print(add_length)
        if add_length >0:
            x = x[:,:-add_length]
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class SINCOS(nn.Module):
    def __init__(self):
        super(SINCOS, self).__init__()
        #self.embed_dim = embed_dim
        # self.pos_embed = self.get_2d_sincos_pos_embed(embed_dim, 8)

    def get_1d_sincos_pos_embed_from_grid(self,embed_dim, pos, device):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float, device=device)
        omega = omega / (embed_dim / 2.)
        omega = 1. / (10000**omega)  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = torch.sin(out) # (M, D/2)
        emb_cos = torch.cos(out) # (M, D/2)

        emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
        return emb

    def get_2d_sincos_pos_embed_from_grid(self,embed_dim, grid, device):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], device)  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], device)  # (H*W, D/2)

        emb = torch.cat([emb_h, emb_w], dim=1) # (H*W, D)
        return emb

    def get_2d_sincos_pos_embed(self,embed_dim, grid_size_h, grid_size_w, device='cpu', cls_token=False):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """

        grid_h = torch.arange(grid_size_h, dtype=torch.float, device=device)
        grid_w = torch.arange(grid_size_w, dtype=torch.float, device=device)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij') 
        grid = torch.stack([grid[1],grid[0]], dim=0)

        grid = grid.reshape(2, 1, grid_size_h, grid_size_w)
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid, device)

        if cls_token:
            pos_embed = torch.cat([torch.zeros(1, embed_dim, device=device), pos_embed], dim=0)
        return pos_embed

    def forward_single(self,x,pos=None,C=None):
        pos_all,pos = pos[0],pos[1:]

        pos = torch.tensor([ _pos[1]*pos_all[0]+_pos[0] for _pos in pos],device=x.device)

        pos_embed = self.get_2d_sincos_pos_embed(C, int(pos_all[1]),int(pos_all[0]),device=x.device)
        
        pos_embed = pos_embed[pos]

        x = x + pos_embed.unsqueeze(0)

        return x

    def forward(self, x, pos=None):
        B,N,C = x.shape

        if pos.size(0) == 1:
            pos = pos[0]

        if len(pos.shape) == 3:
            for i,_pos in enumerate(pos):
                x[i] = self.forward_single(x[i],_pos,C)
        else:
            x = self.forward_single(x,pos,C)
        
        return x

def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

def get_relative_positions(seq_len: int, device='cpu') -> torch.tensor:
    x = torch.arange(seq_len,device=device)[None, :]
    y = torch.arange(seq_len,device=device)[:, None]
    return x - y

def get_alibi_bias_fn(m=None,pos=None,cls_token=False,n_heads=None,device='cpu'):

    pos_all,pos = pos[0],pos[1:]
    pos = torch.tensor([ _pos[1]*pos_all[0]+_pos[0] for _pos in pos],device=device)

    rel_pos = get_relative_positions(int(pos_all[0]*pos_all[1]),device=device)
    rel_pos = rel_pos[pos]
    rel_pos = rel_pos[:,pos]

    if cls_token:
        N = rel_pos.size(-1)
        bias = torch.zeros((1, m.size(0), N+1, N+1), device=device)
        bias[0, :, 1:, 1:] = (rel_pos * m).unsqueeze(0)
    else:
        bias = (rel_pos * m).unsqueeze(0)

    return bias

def get_alibi_bias(m=None,pos=None,cls_token=False,n_heads=None,device='cpu'):
    if pos.size(0) == 1:
        pos = pos[0]
    
    if m is None:
        assert n_heads is not None
        m = get_alibi_slope(n_heads).to(device)
    if len(m.shape) == 1:
        m = m.unsqueeze(-1).unsqueeze(-1)

    # 处理batch
    if len(pos.shape) == 3:
        B,H,N = pos.size(0),m.size(0),pos.size(1)-1
        if cls_token:
            N += 1
        bias = torch.zeros((B,H,N,N),device=device)
        for i,_pos in enumerate(pos):
            bias[i] = get_alibi_bias_fn(m,_pos,cls_token,n_heads,device=device)
    else:
        return get_alibi_bias_fn(m,pos,cls_token,n_heads,device=device)


#################################################################################
#                                 NTK Operations     FIT                        #
#################################################################################

def find_correction_factor(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base)) #Inverse dim formula to find number of rotations

def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(find_correction_factor(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_factor(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1) #Clamp values just in case

def linear_ramp_mask(min, max, dim):
    if (min == max):
        max += 0.001 #Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def find_newbase_ntk(dim, base=10000, scale=1):
    # Base change formula
    return base * scale ** (dim / (dim-2))

def get_mscale(scale=torch.Tensor):
    # if scale <= 1:
    #     return 1.0
    # return 0.1 * math.log(scale) + 1.0
    return torch.where(scale <= 1., torch.tensor(1.0), 0.1 * torch.log(scale) + 1.0)

def get_proportion(L_test, L_train):
    L_test = L_test * 2
    return torch.where(torch.tensor(L_test/L_train) <= 1., torch.tensor(1.0), torch.sqrt(torch.log(torch.tensor(L_test))/torch.log(torch.tensor(L_train))))
    # return torch.sqrt(torch.log(torch.tensor(L_test))/torch.log(torch.tensor(L_train)))

#################################################################################
#                                 Rotate Q or K                                 #
#################################################################################

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

#################################################################################
#                               Core Vision RoPE                                #
#################################################################################
class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,  # embed dimension for each head
        custom_freqs: str = 'normal',
        theta: int = 10000,
        online_rope: bool = False,
        max_cached_len: int = 256,
        max_pe_len_h: Optional[int] = None,
        max_pe_len_w: Optional[int] = None,
        decouple: bool = False,
        ori_max_pe_len: Optional[int] = None,
    ):
        super().__init__()
        
        dim = head_dim // 2
        assert dim % 2 == 0 # accually, this is important
        self.dim = dim
        self.custom_freqs = custom_freqs.lower()
        self.theta = theta
        self.decouple = decouple
        self.ori_max_pe_len = ori_max_pe_len
        
        self.custom_freqs = custom_freqs.lower()
        if not online_rope:
            if self.custom_freqs == 'normal':
                freqs_h = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
                freqs_w = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
            else:
                if decouple:
                    freqs_h = self.get_1d_rope_freqs(theta, dim, max_pe_len_h, ori_max_pe_len)
                    freqs_w = self.get_1d_rope_freqs(theta, dim, max_pe_len_w, ori_max_pe_len)
                else:
                    max_pe_len = max(max_pe_len_h, max_pe_len_w)
                    freqs_h = self.get_1d_rope_freqs(theta, dim, max_pe_len, ori_max_pe_len)
                    freqs_w = self.get_1d_rope_freqs(theta, dim, max_pe_len, ori_max_pe_len)

                attn_factor = 1.0
                scale = torch.clamp_min(torch.tensor(max(max_pe_len_h, max_pe_len_w)) / ori_max_pe_len, 1.0)   # dynamic scale
                self.mscale = get_mscale(scale).to(scale) * attn_factor # Get n-d magnitude scaling corrected for interpolation
                self.proportion1 = get_proportion(max(max_pe_len_h, max_pe_len_w), ori_max_pe_len)
                self.proportion2 = get_proportion(max_pe_len_h * max_pe_len_w, ori_max_pe_len ** 2)
            
            self.register_buffer('freqs_h', freqs_h, persistent=False)        
            self.register_buffer('freqs_w', freqs_w, persistent=False)        
            
            freqs_h_cached = torch.einsum('..., f -> ... f', torch.arange(max_cached_len), self.freqs_h)
            freqs_h_cached = repeat(freqs_h_cached, '... n -> ... (n r)', r = 2)
            self.register_buffer('freqs_h_cached', freqs_h_cached, persistent=False) 
            freqs_w_cached = torch.einsum('..., f -> ... f', torch.arange(max_cached_len), self.freqs_w)
            freqs_w_cached = repeat(freqs_w_cached, '... n -> ... (n r)', r = 2)
            self.register_buffer('freqs_w_cached', freqs_w_cached, persistent=False) 
        
    def get_1d_rope_freqs(self, theta, dim, max_pe_len, ori_max_pe_len):
        # scaling operations for extrapolation
        #assert isinstance(ori_max_pe_len, int)
        # scale = max_pe_len / ori_max_pe_len
        if not isinstance(max_pe_len, torch.Tensor):
            max_pe_len = torch.tensor(max_pe_len)
            
        if ori_max_pe_len is None:
            ori_max_pe_len = max_pe_len.clone()
            
        scale = torch.clamp_min(max_pe_len / ori_max_pe_len, 1.0)   # dynamic scale
            
        if self.custom_freqs == 'linear': # equal to position interpolation
            freqs = 1. / torch.einsum('..., f -> ... f', scale, theta ** (torch.arange(0, dim, 2).float() / dim))
        elif self.custom_freqs == 'ntk-aware' or self.custom_freqs == 'ntk-aware-pro1' or self.custom_freqs == 'ntk-aware-pro2':
            freqs = 1. / torch.pow(
                find_newbase_ntk(dim, theta, scale).view(-1, 1), 
                (torch.arange(0, dim, 2).to(scale).float() / dim)
            )
            if len(freqs.shape) > 2:
                freqs = freqs.squeeze()
            #print(freqs.shape)
            #
        elif self.custom_freqs == 'ntk-by-parts':
            #Interpolation constants found experimentally for LLaMA (might not be totally optimal though)
            #Do not change unless there is a good reason for doing so!
            beta_0 = 1.25
            beta_1 = 0.75
            gamma_0 = 16
            gamma_1 = 2
            ntk_factor = 1
            extrapolation_factor = 1

            #Three RoPE extrapolation/interpolation methods
            freqs_base = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            freqs_linear = 1.0 / torch.einsum('..., f -> ... f', scale, (theta ** (torch.arange(0, dim, 2).to(scale).float() / dim)))
            freqs_ntk = 1. / torch.pow(
                find_newbase_ntk(dim, theta, scale).view(-1, 1), 
                (torch.arange(0, dim, 2).to(scale).float() / dim)
            ).squeeze()
            
            #Combine NTK and Linear
            low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
            freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(scale)) * ntk_factor
            freqs = freqs_linear * (1 - freqs_mask) + freqs_ntk * freqs_mask
            
            #Combine Extrapolation and NTK and Linear
            low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
            freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(scale)) * extrapolation_factor
            freqs = freqs * (1 - freqs_mask) + freqs_base * freqs_mask
            
        elif self.custom_freqs == 'yarn':
            #Interpolation constants found experimentally for LLaMA (might not be totally optimal though)
            #Do not change unless there is a good reason for doing so!
            beta_fast = 32
            beta_slow = 1
            extrapolation_factor = 1
            
            freqs_extrapolation = 1.0 / (theta ** (torch.arange(0, dim, 2).to(scale).float() / dim))
            freqs_interpolation = 1.0 / torch.einsum('..., f -> ... f', scale, (theta ** (torch.arange(0, dim, 2).to(scale).float() / dim)))

            low, high = find_correction_range(beta_fast, beta_slow, dim, theta, ori_max_pe_len)
            freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(scale).float()) * extrapolation_factor # Get n-d rotational scaling corrected for extrapolation
            freqs = freqs_interpolation * (1 - freqs_mask) + freqs_extrapolation * freqs_mask            
        else:
            raise ValueError(f'Unknown modality {self.custom_freqs}. Only support normal, linear, ntk-aware, ntk-by-parts, yarn!')
        return freqs

    def online_get_2d_rope_from_grid(self, grid, size):
        '''
        grid: (B, 2, N)
            N = H * W
            the first dimension represents width, and the second reprensents height
            e.g.,   [0. 1. 2. 3. 0. 1. 2. 3. 0. 1. 2. 3.]
                    [0. 0. 0. 0. 1. 1. 1. 1. 2. 2. 2. 2.]
        size: (B, 1, 2), h goes first and w goes last
        '''
        size = size.squeeze(1)   # (B, 1, 2) -> (B, 2)
        if self.decouple:
            size_h = size[:, 0]
            size_w = size[:, 1]
            freqs_h = self.get_1d_rope_freqs(self.theta, self.dim, size_h, self.ori_max_pe_len)
            freqs_w = self.get_1d_rope_freqs(self.theta, self.dim, size_w, self.ori_max_pe_len)
        else:
            size_max = torch.max(size[:, 0], size[:, 1])
            freqs_h = self.get_1d_rope_freqs(self.theta, self.dim, size_max, self.ori_max_pe_len)
            freqs_w = self.get_1d_rope_freqs(self.theta, self.dim, size_max, self.ori_max_pe_len)
        freqs_w = grid[:, 0][..., None] * freqs_w[:, None, :]
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)
        
        freqs_h = grid[:, 1][..., None] * freqs_h[:, None, :]
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)
        
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)   # (B, N, D)
        
        if self.custom_freqs == 'yarn':
            freqs_cos = freqs.cos() * self.mscale[:, None, None]
            freqs_sin = freqs.sin() * self.mscale[:, None, None]
        elif self.custom_freqs == 'ntk-aware-pro1':
            freqs_cos = freqs.cos() * self.proportion1[:, None, None]
            freqs_sin = freqs.sin() * self.proportion1[:, None, None]
        elif self.custom_freqs == 'ntk-aware-pro2':
            freqs_cos = freqs.cos() * self.proportion2[:, None, None]
            freqs_sin = freqs.sin() * self.proportion2[:, None, None]
        else:
            freqs_cos = freqs.cos()
            freqs_sin = freqs.sin()
            
        return freqs_cos, freqs_sin  

    @lru_cache()
    def get_2d_rope_from_grid(self, grid):
        '''
        grid: (B, 2, N)
            N = H * W
            the first dimension represents width, and the second reprensents height
            e.g.,   [0. 1. 2. 3. 0. 1. 2. 3. 0. 1. 2. 3.]
                    [0. 0. 0. 0. 1. 1. 1. 1. 2. 2. 2. 2.]
        '''  
        freqs_w = torch.einsum('..., f -> ... f', grid[:, 0], self.freqs_w)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)
        
        freqs_h = torch.einsum('..., f -> ... f', grid[:, 1], self.freqs_h)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)
        
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)   # (B, N, D)
        
        if self.custom_freqs == 'yarn':
            freqs_cos = freqs.cos() * self.mscale
            freqs_sin = freqs.sin() * self.mscale
        elif self.custom_freqs == 'ntk-aware-pro1':
            freqs_cos = freqs.cos() * self.proportion1
            freqs_sin = freqs.sin() * self.proportion1
        elif self.custom_freqs == 'ntk-aware-pro2':
            freqs_cos = freqs.cos() * self.proportion2
            freqs_sin = freqs.sin() * self.proportion2
        else:
            freqs_cos = freqs.cos()
            freqs_sin = freqs.sin()

        return freqs_cos, freqs_sin
    
    @lru_cache()
    def get_cached_2d_rope_from_grid(self, grid: torch.Tensor):
        '''
        grid: (B, 2, N)
            N = H * W
            the first dimension represents width, and the second reprensents height
            e.g.,   [0. 1. 2. 3. 0. 1. 2. 3. 0. 1. 2. 3.]
                    [0. 0. 0. 0. 1. 1. 1. 1. 2. 2. 2. 2.]
        '''  
        freqs_w, freqs_h = self.freqs_w_cached[grid[:, 0]], self.freqs_h_cached[grid[:, 1]]
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)   # (B, N, D)
        
        if self.custom_freqs == 'yarn':
            freqs_cos = freqs.cos() * self.mscale
            freqs_sin = freqs.sin() * self.mscale
        elif self.custom_freqs == 'ntk-aware-pro1':
            freqs_cos = freqs.cos() * self.proportion1
            freqs_sin = freqs.sin() * self.proportion1
        elif self.custom_freqs == 'ntk-aware-pro2':
            freqs_cos = freqs.cos() * self.proportion2
            freqs_sin = freqs.sin() * self.proportion2
        else:
            freqs_cos = freqs.cos()
            freqs_sin = freqs.sin()
        
        return freqs_cos, freqs_sin

    @lru_cache()
    def get_cached_21d_rope_from_grid(self, grid: torch.Tensor): # for 3d rope formulation 2 !
        '''
        grid: (B, 3, N)
            N = H * W * T
            the first dimension represents width, and the second reprensents height, and the third reprensents time
            e.g.,   [0. 1. 2. 3. 0. 1. 2. 3. 0. 1. 2. 3.]
                    [0. 0. 0. 0. 1. 1. 1. 1. 2. 2. 2. 2.]
                    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        '''   
        freqs_w, freqs_h = self.freqs_w_cached[grid[:, 0]+grid[:, 2]], self.freqs_h_cached[grid[:, 1]+grid[:, 2]]
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)   # (B, N, D)
        
        if self.custom_freqs == 'yarn':
            freqs_cos = freqs.cos() * self.mscale
            freqs_sin = freqs.sin() * self.mscale
        elif self.custom_freqs == 'ntk-aware-pro1':
            freqs_cos = freqs.cos() * self.proportion1
            freqs_sin = freqs.sin() * self.proportion1
        elif self.custom_freqs == 'ntk-aware-pro2':
            freqs_cos = freqs.cos() * self.proportion2
            freqs_sin = freqs.sin() * self.proportion2
        else:
            freqs_cos = freqs.cos()
            freqs_sin = freqs.sin()
        
        return freqs_cos, freqs_sin

    def forward(self, x, grid): 
        '''
        x: (B, n_head, N, D)
        grid: (B, 2, N)
        '''
        # freqs_cos, freqs_sin = self.get_2d_rope_from_grid(grid)
        # freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        # using cache to accelerate, this is the same with the above codes:
        freqs_cos, freqs_sin = self.get_cached_2d_rope_from_grid(grid)
        freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        return  x * freqs_cos + rotate_half(x) * freqs_sin

    def get_grid(self,batch_size, h=14, w=14, device="cuda"):
        # 创建网格坐标
        grid_h = torch.arange(h, device=device)
        grid_w = torch.arange(w, device=device)
        
        # 展开成序列
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        grid_w = grid_w.flatten()
        grid_h = grid_h.flatten()
        
        # 堆叠成最终网格 
        grid = torch.stack([grid_w, grid_h], dim=0)[None]  # (1, 2, H*W)
        
        # 如果batch_size>1,重复到对应batch size
        grid = grid.repeat(batch_size, 1, 1)  # (B, 2, H*W)
        
        return grid

    # dropout后只保留部分token的情况 
    def apply_rope_with_dropout(self,x, h,w,keep_indices=None):
        """
        x: (B, num_heads, L, head_dim)
        keep_indices: (B, num_kept) - 每个batch中要保留的token索引
        """
        B = x.size(0)
        grid = self.get_grid(B,h, w, x.device)  # (B, 2, 196)
        
        if keep_indices is not None:
            # 为每个batch选择对应的grid位置
            grid_kept = torch.stack([
                grid[b, :, idx] for b, idx in enumerate(keep_indices)
            ])  # (B, 2, num_kept)
        else:
            grid_kept = grid

        # 应用RoPE
        freqs_cos, freqs_sin = self.get_cached_2d_rope_from_grid(grid_kept)

        return freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

#################################################################################
#               ROPE VisionLLaMA                 #
#################################################################################
def precompute_freqs_cis_2d(dim: int, end: int, theta: float = 10000.0, scale=1.0, use_cls=False):
    H = int( end**0.5 )
    # assert  H * H == end
    flat_patch_pos = torch.arange(0 if not use_cls else -1, end) # N = end
    x_pos = flat_patch_pos % H # N
    y_pos = flat_patch_pos // H # N
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float() # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float() # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1) # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(end if not use_cls else end + 1, -1)
    # we need to think how to implement this for multi heads.
    # freqs_cis = torch.cat([x_cis, y_cis], dim=-1) # N, Hc/2
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # x: B N H Hc/2
    # freqs_cis:  N, H*Hc/2 or  N Hc/2
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if freqs_cis.shape[-1] == x.shape[-1]:
        shape = [1 if i == 2 or i == 0 else d for i, d in enumerate(x.shape)]  # 1, N, 1, Hc/2
    else:
        shape = [d if i != 0 else 1 for i, d in enumerate(x.shape)] # 1, N, H, Hc/2
        # B, N, Hc/2
    return freqs_cis.view(*shape)

def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq : B N H Hc
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # B N H Hc/2
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # B, N, H, Hc
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)