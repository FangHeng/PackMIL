import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.attention import SDPBackend, sdpa_kernel

from einops import repeat, rearrange
from functools import partial

from modules.emb_position import *
from modules.rrt import RRTEncoder
from modules.vit_mil import Attention
from modules.nystrom_attention import NystromAttention
from modules.utils import *


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper
class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim=-1)
        return normed * self.scale * self.gamma


class AttentionPool(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = nn.LayerNorm(dim, bias=False)

        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(
            self,
            q,
            kv=None,
            mask=None,
            attn_mask=None
    ):
        q = self.norm(q)
        kv_input = default(kv, q)

        qkv = (self.to_q(q), *self.to_kv(kv_input).chunk(2, dim=-1))

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        dots = torch.matmul(q, k.transpose(-1, -2))

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, deterministic=True, attn_type='naive', mil_bias=False,dropout=0., sdpa_type='torch', norm=True,res=False):
        super().__init__()
        self.norm = norm_layer(dim, bias=mil_bias) if norm else nn.Identity()
        if attn_type == 'naive':
            self.attn = Attention(
                dim=dim,
                num_heads=8,
                attn_drop=dropout,
                deterministic=deterministic,
                sdpa_type=sdpa_type,
                residual=res,
            )
        elif attn_type == 'ntrans':
            self.attn = NystromAttention(
                dim = dim,
                dim_head=dim//8,
                heads = 8,
                num_landmarks = dim//2,    # number of landmarks
                pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
                residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
                dropout=dropout
                )
        else:
            raise NotImplementedError

    def forward(self, x, attn_mask=None, need_attn=False, need_v=False, no_norm=False):
        if need_attn:
            z, attn, v = self.attn(self.norm(x), return_attn=need_attn, attn_mask=attn_mask)
            x = x + z
            if need_v:
                return x, attn, v
            else:
                return x, attn
        else:
            x = x + self.attn(self.norm(x), attn_mask=attn_mask)
            return x


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=True, norm=None,
                 fc_norm_bn=False):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        self.mil_norm = norm
        if norm == 'bn':
            if fc_norm_bn:
                self.norm = nn.BatchNorm1d(input_size)
            else:
                self.norm = nn.Identity()
        elif norm == 'ln':
            self.norm = nn.LayerNorm(input_size)
        else:
            self.norm = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, m_indices, no_norm=False, attn_mask=None, is_images=None,
                ban_norm=False):  # B x N x K, B x N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats)  # N x Q, unsorted

        # handle multiple classes without for loop
        # _, m_indices = torch.sort(c, 1, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        if attn_mask is not None:
            B, N, D = feats.shape
            B, M, C = m_indices.shape

            # Expand indices to include the batch and feature dimensions
            expanded_indices = m_indices.view(B, M * C).unsqueeze(-1).expand(-1, -1, D)  # (B, M*C, D)

            # Gather the features
            m_feats = torch.gather(feats, dim=1, index=expanded_indices)  # (B, M*C, D)
            # m_feats = m_feats.view(B, M, C, D)
        else:
            m_feats = torch.gather(feats, dim=1, index=m_indices.unsqueeze(-1).expand(-1, -1, feats.size(
                -1)))  # select critical instances,

        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = Q @ q_max.transpose(-2, -1)
        A = A / torch.sqrt(torch.tensor(Q.shape[-1], dtype=A.dtype, device=device))

        if attn_mask is not None:
            A = A.masked_fill(rearrange(~attn_mask, 'b i j c -> b j (i c)'), -torch.finfo(A.dtype).max)

        if no_norm:
            _A = A
        A = F.softmax(A, -2)  # normalize attention scores, A in shape N x C,
        if not no_norm:
            _A = A
        bag_feat = A.transpose(-2, -1) @ V

        if attn_mask is not None:
            bag_feat = bag_feat.reshape(B * M, C, D)

        if ban_norm:
            pass
        elif self.mil_norm == 'bn':
            if is_images is not None:
                bag_feat[is_images] = self.norm(bag_feat[is_images].transpose(-1, -2)).transpose(-1, -2)
            else:
                bag_feat = torch.transpose(bag_feat, -1, -2)
                bag_feat = self.norm(bag_feat)
                bag_feat = torch.transpose(bag_feat, -1, -2)
        else:
            if is_images is not None:
                bag_feat[is_images] = self.norm(bag_feat[is_images].type(torch.float32)).type(bag_feat[is_images].dtype)
            else:
                bag_feat = self.norm(bag_feat)

        C = self.fcc(bag_feat)  # B x C x 1
        C = C.squeeze(-1)  # B x C

        return C, _A, bag_feat


class SAttention(nn.Module):
    def __init__(self, inner_dim=512, mil_bias=False, n_layers=1, pos=None, pool='cls_token',attn_type='naive', attn_dropout=0., deterministic=True, sdpa_type='torch', fc_norm=True, vit_norm=True,attn_res=False,**kwargs):
        super(SAttention, self).__init__()
        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=inner_dim)
        else:
            raise NotImplementedError
        self.pos = pos
        if vit_norm:
            if fc_norm:
                self.norm = nn.Identity()
                self.fc_norm = nn.LayerNorm(inner_dim, bias=mil_bias)
            else:
                self.norm = nn.LayerNorm(inner_dim, bias=mil_bias)
                self.fc_norm = nn.Identity()
        else:
            self.norm = nn.Identity()
            self.fc_norm = nn.Identity()

        self.attn_type = attn_type

        self.layer1 = TransLayer(dim=inner_dim, attn_type=attn_type, mil_bias=mil_bias, dropout=attn_dropout,deterministic=deterministic, sdpa_type=sdpa_type, norm=vit_norm,res=attn_res)
        if n_layers >= 2:
            self.layers = [TransLayer(dim=inner_dim, attn_type=attn_type, mil_bias=mil_bias, dropout=attn_dropout,deterministic=deterministic, sdpa_type=sdpa_type, norm=vit_norm,res=attn_res) for i in range(n_layers - 1)]
            self.layers = nn.Sequential(*self.layers)
        else:
            self.layers = None

        self.cls_token = None
        self.pool = pool
        if pool == 'cls_token':
            self.cls_token = nn.Parameter(torch.randn(1, 1, inner_dim))
            nn.init.normal_(self.cls_token, std=1e-6)
        elif pool == 'attn':
            self.attn_pool_queries = nn.Parameter(torch.randn(inner_dim))
            self.pool_fn = AttentionPool(dim=inner_dim, heads=8)
        else:
            raise NotImplementedError

    def insert_cls_token(self, x, cls_token_mask=None):
        num_cls_tokens = cls_token_mask.sum(dim=1)  # [batch]
        cls_tokens = repeat(self.cls_token, '1 n d -> (b n) d', b=num_cls_tokens.sum().item())

        new_x = torch.zeros(
            (x.shape[0], cls_token_mask.shape[1], x.shape[2]),
            dtype=x.dtype,
            device=x.device
        )

        cls_token_idx = 0
        for i in range(x.shape[0]):
            insert_positions = torch.where(cls_token_mask[i])[0]

            current_num_cls = num_cls_tokens[i].item()
            current_cls_tokens = cls_tokens[cls_token_idx:cls_token_idx + current_num_cls]
            cls_token_idx += current_num_cls

            src_pos = 0
            dst_pos = 0

            for k, insert_pos in enumerate(insert_positions):
                num_tokens_before = insert_pos - dst_pos

                if num_tokens_before > 0:
                    new_x[i, dst_pos:insert_pos] = x[i, src_pos:src_pos + num_tokens_before]
                    src_pos += num_tokens_before
                    dst_pos += num_tokens_before

                new_x[i, insert_pos] = current_cls_tokens[k]
                dst_pos += 1

            if src_pos < x.shape[1]:
                remaining = new_x.shape[1] - dst_pos
                if src_pos + remaining > x.shape[1]:
                    remaining = x.shape[1] - src_pos

                new_x[i, dst_pos:dst_pos + remaining] = x[i, src_pos:src_pos + remaining]

        return new_x

    def forward_ntrans(self,x, pack_args=None, return_attn=False, return_feat=False, pos=None,**kwargs):
        batch, num_patches, C = x.shape
        attn_mask = None
        cls_token_mask = None
        key_pad_mask = None
        if pack_args:
            attn_mask = default(pack_args['attn_mask'], None)
            num_images = pack_args['num_images']
            batched_image_ids = pack_args['batched_image_ids']
            key_pad_mask = pack_args['key_pad_mask']
            cls_token_mask = pack_args['cls_token_mask']
            batched_feat_ids_1 = pack_args['batched_image_ids_1']

        if self.cls_token is not None:
            # cls_token
            if cls_token_mask is None:
                cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=batch)
                x = torch.cat((cls_tokens, x), dim=1)
            else:
                x = self.insert_cls_token(x, cls_token_mask)

        attn = []
        # translayer1
        if return_attn:
            raise NotImplementedError

        if cls_token_mask is not None:
            for i in range(x.shape[0]):
                for k in range(num_images[i]):
                    _mask= batched_feat_ids_1[i] == k + 1
                    _mask_pos = _mask * (~cls_token_mask[i]).bool()
                    x[i, _mask] = self.layer1(x[i, _mask].unsqueeze(0)).squeeze(0)
                    x[i, _mask_pos] = self.pos_embedding(x[i, _mask_pos])
                    for _layer in self.layers:
                        x[i, _mask] = _layer(x[i, _mask].unsqueeze(0)).squeeze(0)
        else:
            x = self.layer1(x)
            if key_pad_mask is not None:
                x = x.masked_fill(key_pad_mask.unsqueeze(-1), 0.)
            x[:, 1:, :] = self.pos_embedding(x[:, 1:, :])
            if self.layers:
                for _layer in self.layers:
                    x = _layer(x)
                    if key_pad_mask is not None:
                        x = x.masked_fill(key_pad_mask.unsqueeze(-1), 0.)

        #---->cls_token
        x = self.norm(x)

        if cls_token_mask is None:
            if return_feat:
                x = self.fc_norm(x)
                return x[:, 0], x[:, 1:]
            return self.fc_norm(x[:, 0])
        else:
            return self.fc_norm(x[cls_token_mask])

    def forward(self, x, pack_args=None, return_attn=False, return_feat=False, pos=None,**kwargs):
        if self.attn_type == 'ntrans':
            return self.forward_ntrans(x, pack_args, return_attn, return_feat, pos)
        batch, num_patches, C = x.shape
        attn_mask = None
        cls_token_mask = None
        key_pad_mask = None
        if pack_args:
            attn_mask = default(pack_args['attn_mask'], None)
            num_images = pack_args['num_images']
            batched_image_ids = pack_args['batched_image_ids']
            key_pad_mask = pack_args['key_pad_mask']
            cls_token_mask = pack_args['cls_token_mask']
            batched_feat_ids_1 = pack_args['batched_image_ids_1']

        if self.cls_token is not None:
            # cls_token
            if cls_token_mask is None:
                cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=batch)
                x = torch.cat((cls_tokens, x), dim=1)
            else:
                x = self.insert_cls_token(x, cls_token_mask)
        attn = []
        # translayer1
        if return_attn:
            if self.attn_type == 'ntrans':
                raise NotImplementedError
            x, _attn, v = self.layer1(x, attn_mask=attn_mask, need_attn=True, need_v=True)
            attn.append(_attn.clone())
            
        else:
            if self.attn_type == 'ntrans':
                if cls_token_mask is not None:
                    for i in range(x.shape[0]):
                        for k in range(num_images[i]):
                            _mask= batched_feat_ids_1[i] == k + 1
                            x[i, _mask] = self.layer1(x[i, _mask].unsqueeze(0)).squeeze(0)
                else:
                    x = self.layer1(x)
                    if key_pad_mask is not None:
                        x = x.masked_fill(key_pad_mask.unsqueeze(-1), 0.)

            else:
                x = self.layer1(x, attn_mask=attn_mask)

        # add pos embedding
        if cls_token_mask is None:
            if self.pool == 'cls_token':
                x[:, 1:, :] = self.pos_embedding(x[:, 1:, :])
        else:
            for i in range(x.shape[0]):
                for k in range(num_images[i]):
                    #_mask_old = batched_feat_ids_1[i] == k + 1
                    _mask = ((batched_feat_ids_1[i] == k + 1) * (~cls_token_mask[i])).bool()
                    x[i, _mask] = self.pos_embedding(x[i, _mask])

        # translayer more
        if self.layers:
            for _layer in self.layers:
                if return_attn:
                    x, _attn, _ = _layer(x, attn_mask=attn_mask, need_attn=True, need_v=True)
                    attn.append(_attn.clone())
                else:
                    if self.attn_type == 'ntrans':
                        if cls_token_mask is not None:
                            for i in range(x.shape[0]):
                                for k in range(num_images[i]):
                                    _mask= batched_feat_ids_1[i] == k + 1
                                    x[i, _mask] = _layer(x[i, _mask].unsqueeze(0)).squeeze(0)
                        else:
                            x = _layer(x)
                            if key_pad_mask is not None:
                                x = x.masked_fill(key_pad_mask.unsqueeze(-1), 0.)
                    else:
                        x = _layer(x, attn_mask=attn_mask)

        # #---->cls_token
        x = self.norm(x)

        # -----> attn pool
        if self.pool == 'attn':
            if pack_args:
                arange = partial(torch.arange, device=x.device)
                max_queries = num_images.amax().item()

                queries = repeat(self.attn_pool_queries, 'd -> b n d', n=max_queries, b=x.shape[0])

                # attention pool mask

                image_id_arange = arange(max_queries)

                attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_image_ids, 'b j -> b 1 j')

                attn_pool_mask = attn_pool_mask & rearrange(~key_pad_mask, 'b j -> b 1 j')

                attn_pool_mask = rearrange(attn_pool_mask, 'b i j -> b 1 i j')

                # attention pool

                x = self.pool_fn(queries, kv=x, attn_mask=attn_pool_mask) + queries

                x = rearrange(x, 'b n d -> (b n) d')

                # each batch element may not have same amount of images

                is_images = image_id_arange < rearrange(num_images, 'b -> b 1')
                is_images = rearrange(is_images, 'b n -> (b n)')

                x = x[is_images]

            else:
                queries = repeat(self.attn_pool_queries, 'd -> b 1 d', b=x.shape[0])
                x = self.pool_fn(queries, kv=x) + queries

            return self.fc_norm(x)

        elif self.pool == 'cls_token':
            if cls_token_mask is None:
                if return_feat:
                    x = self.fc_norm(x)
                    return x[:, 0], x[:, 1:]
                return self.fc_norm(x[:, 0])
            else:
                return self.fc_norm(x[cls_token_mask])


class DAttention(nn.Module):
    def __init__(self, inner_dim=512, n_classes=0, mil_bias=False, da_gated=False, cls_norm=None, fc_norm_bn=False,use_deterministic_softmax=False, mil_norm=None, **kwargs):
        super(DAttention, self).__init__()

        self.L = inner_dim  # 512
        self.D = 128  # 128
        self.K = 1
        self.da_gated = da_gated
        cls_norm = mil_norm if cls_norm is None else cls_norm
        if da_gated:
            self.attention_a = [
                nn.Linear(self.L, self.D, bias=mil_bias),
            ]
            self.attention_a += [nn.Tanh()]

            self.attention_b = [nn.Linear(self.L, self.D, bias=mil_bias),
                                nn.Sigmoid()]

            self.attention_a = nn.Sequential(*self.attention_a)
            self.attention_b = nn.Sequential(*self.attention_b)
            self.attention_c = nn.Linear(self.D, self.K, bias=mil_bias)
        else:
            self.attention = nn.Sequential(
                nn.Linear(self.L, self.D, bias=mil_bias),
                nn.Tanh(),
                nn.Linear(self.D, self.K, bias=mil_bias)
            )

        if cls_norm == 'bn':
            if fc_norm_bn:
                self.norm1 = nn.BatchNorm1d(self.L * self.K)
            else:
                self.norm1 = nn.Identity()
        elif cls_norm == 'ln':
            self.norm1 = nn.LayerNorm(self.L * self.K, bias=mil_bias)
        else:
            self.norm1 = nn.Identity()

        self.use_deterministic_softmax = use_deterministic_softmax

    def forward(self, x, pack_args=None, return_attn=False, pos=None, ban_norm=False,**kwargs):
        if self.da_gated:
            A = self.attention_a(x)
            b = self.attention_b(x)
            A = A.mul(b)
            A = self.attention_c(A)
        else:
            A = self.attention(x)  # B N K
        A = torch.transpose(A, -1, -2)  # KxN
        if pack_args is not None:
            num_feats = pack_args['num_images']
            batched_feat_ids = pack_args['batched_image_ids']
            key_pad_mask = pack_args['key_pad_mask']

            if batched_feat_ids is not None:
                max_queries = num_feats.amax().item()
                arange = partial(torch.arange, device=x.device)
                # attention pool mask
                image_id_arange = arange(max_queries)
                attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_feat_ids, 'b j -> b 1 j')
                attn_pool_mask = attn_pool_mask & rearrange(~key_pad_mask, 'b j -> b 1 j')

                A = repeat(A, 'b 1 n -> b m n', m=max_queries)
                A = A.masked_fill(~attn_pool_mask, -torch.finfo(A.dtype).max)
            else:
                key_pad_mask = key_pad_mask.unsqueeze(1)
                A = A.masked_fill(key_pad_mask, -torch.finfo(A.dtype).max)

        A = F.softmax(A, dim=-1)

        x = torch.einsum('b k n, b n d -> b k d', A, x).squeeze(1)
        if pack_args is not None:
            if batched_feat_ids is not None:
                if len(x.shape) > 2:
                    x = rearrange(x, 'b n d -> (b n) d')
                # each batch element may not have same amount of images
                is_images = image_id_arange < rearrange(num_feats, 'b -> b 1')
                is_images = rearrange(is_images, 'b n -> (b n)')
                x = x[is_images]

        if pack_args is not None and ban_norm:
            pass
        else:
            x = self.norm1(x).squeeze(1)

        if return_attn:
            return x, A
        else:
            return x


class RRTMIL(nn.Module):
    def __init__(self, inner_dim=512, n_classes=0, mil_bias=False, da_gated=False, mil_norm=None, fc_norm_bn=False,
                 pos_pos=0, pos='none', peg_k=7, attn='rmsa', region_num=8, n_layers=2, n_heads=8, drop_path=0.,
                 da_act='relu', trans_dropout=0.1, ffn=False, ffn_act='gelu', mlp_ratio=4., trans_dim=64, epeg=True,
                 min_region_num=0, qkv_bias=True, **kwargs):
        super(RRTMIL, self).__init__()

        self.L = inner_dim  # 512
        self.D = 128  # 128
        self.K = 1
        self.da_gated = da_gated
        if da_gated:
            self.attention_a = [
                nn.Linear(self.L, self.D, bias=mil_bias),
            ]
            if da_act == 'gelu':
                self.attention_a += [nn.GELU()]
            elif da_act == 'relu':
                self.attention_a += [nn.ReLU()]
            elif da_act == 'tanh':
                self.attention_a += [nn.Tanh()]

            self.attention_b = [nn.Linear(self.L, self.D, bias=mil_bias),
                                nn.Sigmoid()]

            self.attention_a = nn.Sequential(*self.attention_a)
            self.attention_b = nn.Sequential(*self.attention_b)
            self.attention_c = nn.Linear(self.D, self.K, bias=mil_bias)
        else:
            self.attention = [
                nn.Linear(self.L, self.D, bias=mil_bias),
            ]
            if da_act == 'gelu':
                self.attention += [nn.GELU()]
            elif da_act == 'relu':
                self.attention += [nn.ReLU()]
            elif da_act == 'tanh':
                self.attention += [nn.Tanh()]
            self.attention += [
                nn.Linear(self.D, self.K, bias=mil_bias)
            ]
            self.attention = nn.Sequential(*self.attention)

        if mil_norm == 'bn':
            if fc_norm_bn:
                self.norm1 = nn.BatchNorm1d(self.L * self.K)
            else:
                self.norm1 = nn.Identity()
        elif mil_norm == 'ln':
            self.norm1 = nn.LayerNorm(self.L * self.K, bias=mil_bias)
        else:
            self.norm1 = nn.Identity()

        self.online_encoder = RRTEncoder(mlp_dim=inner_dim, pos_pos=pos_pos, pos=pos, peg_k=peg_k, attn=attn,
                                         region_num=region_num, n_layers=n_layers, n_heads=n_heads, drop_path=drop_path,
                                         drop_out=trans_dropout, ffn=ffn, ffn_act=ffn_act, mlp_ratio=mlp_ratio,
                                         trans_dim=trans_dim, epeg=epeg, min_region_num=min_region_num,
                                         qkv_bias=qkv_bias, **kwargs)

    def forward(self, x, pack_args=None, return_attn=False, ban_norm=False, pos=None,**kwargs):
        if pack_args is not None:
            num_feats = pack_args['num_images']
            batched_feat_ids = pack_args['batched_image_ids']
            batched_feat_ids_1 = pack_args['batched_image_ids_1']
            key_pad_mask = pack_args['key_pad_mask']

            if num_feats is not None:
                for i in range(x.shape[0]):
                    for k in range(num_feats[i]):
                        _mask = batched_feat_ids_1[i] == k + 1
                        x[i, _mask] = self.online_encoder(x[i, _mask])
            else:
                assert key_pad_mask is not None, "key_pad_mask is required when resuming from pack_args"
                x = apply_function_nonpad(x, key_pad_mask, self.online_encoder)

            if self.da_gated:
                A = self.attention_a(x)
                b = self.attention_b(x)
                A = A.mul(b)
                A = self.attention_c(A)
            else:
                A = self.attention(x)  # B N K
            A = torch.transpose(A, -1, -2)  # KxN

            if batched_feat_ids is not None and batched_feat_ids.sum().item() != 0:
                max_queries = num_feats.amax().item()
                arange = partial(torch.arange, device=x.device)
                # attention pool mask
                image_id_arange = arange(max_queries)
                attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_feat_ids, 'b j -> b 1 j')
                attn_pool_mask = attn_pool_mask & rearrange(~key_pad_mask, 'b j -> b 1 j')
                # attn_pool_mask = rearrange(attn_pool_mask, 'b i j -> b i j')

                A = repeat(A, 'b 1 n -> b m n', m=max_queries)
                A = A.masked_fill(~attn_pool_mask, -torch.finfo(A.dtype).max)
            else:
                key_pad_mask = key_pad_mask.unsqueeze(1)
                A = A.masked_fill(key_pad_mask, -torch.finfo(A.dtype).max)
        else:
            x = self.online_encoder(x)
            if self.da_gated:
                A = self.attention_a(x)
                b = self.attention_b(x)
                A = A.mul(b)
                A = self.attention_c(A)
            else:
                A = self.attention(x)  # B N K
            A = torch.transpose(A, -1, -2)  # KxN

        A = F.softmax(A, dim=-1)  # softmax over N

        x = torch.einsum('b k n, b n d -> b k d', A, x).squeeze(1)
        if pack_args is not None:
            if batched_feat_ids is not None and batched_feat_ids.sum().item() != 0:
                x = rearrange(x, 'b n d -> (b n) d')
                # each batch element may not have same amount of images
                is_images = image_id_arange < rearrange(num_feats, 'b -> b 1')
                is_images = rearrange(is_images, 'b n -> (b n)')
                x = x[is_images]

        if pack_args is None or not ban_norm:
            x = self.norm1(x).squeeze(1)

        if return_attn:
            return x, A
        else:
            return x


class DSAttention(nn.Module):
    def __init__(self, inner_dim=512, mil_bias=False, agg_n_classes=0, mil_norm=None, fc_norm_bn=False, cls_norm=None,
                 **kwargs):
        super(DSAttention, self).__init__()

        self.i_classifier = nn.Sequential(
            nn.Linear(inner_dim, agg_n_classes))

        cls_norm = mil_norm if cls_norm is None else cls_norm

        self.b_classifier = BClassifier(inner_dim, agg_n_classes, norm=cls_norm, fc_norm_bn=fc_norm_bn)

    def forward(self, x, pack_args=None, return_attn=False, return_cam=False, label=None, loss=None, pos=None,ban_norm=False,residual=False):
        bs, ps, _ = x.shape
        classes = self.i_classifier(x)
        if pack_args is None:
            classes, m_indices = torch.max(classes, 1)
            prediction_bag, A, B = self.b_classifier(x, m_indices)

        else:
            num_feats = pack_args.get('num_images',None)
            batched_feat_ids = pack_args.get('batched_image_ids',None)
            key_pad_mask = pack_args.get('key_pad_mask', None)
            no_norm_pad = pack_args.get('no_norm_pad', None)
            if not residual:
                residual = pack_args.get('residual', False)

            if batched_feat_ids is not None:
                max_queries = num_feats.amax().item()
                arange = partial(torch.arange, device=x.device)
                image_id_arange = arange(max_queries)
                # attention pool mask
                attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_feat_ids, 'b j -> b 1 j')
                attn_pool_mask = attn_pool_mask & rearrange(~key_pad_mask, 'b j -> b 1 j')
                attn_pool_mask = repeat(attn_pool_mask, 'b i j -> b i j c', c=classes.shape[-1])

                is_images = image_id_arange < rearrange(num_feats, 'b -> b 1')
                is_images = rearrange(is_images, 'b n -> (b n)')

                classes = repeat(classes, 'b n c -> b m n c', m=max_queries)
                classes = classes.masked_fill(~attn_pool_mask, -torch.finfo(classes.dtype).max)
                classes, m_indices = torch.max(classes, dim=2)
                prediction_bag, A, B = self.b_classifier(
                    x, m_indices,
                    attn_mask=attn_pool_mask,
                    is_images=is_images if no_norm_pad else None,
                    ban_norm=True if residual and ban_norm else False
                )
                classes = rearrange(classes, 'b n d -> (b n) d')

                classes = classes[is_images]
                prediction_bag = prediction_bag[is_images]
                bs = classes.shape[0]
            else:
                classes, m_indices = torch.max(classes, dim=1)
                prediction_bag, A, B = self.b_classifier(x, m_indices)

        if residual:
            return [prediction_bag, classes, B]
        else:
            return [prediction_bag, classes]


def apply_function_nonpad(
        x: torch.Tensor,  # [B, L, in_dim]
        mask: torch.BoolTensor,  # [B, L]
        func: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """
    apply a function to non-padding elements of a tensor
    """
    B, L, D_in = x.shape

    x_nonpad = x[~mask]  # [N, D_in]

    x_nonpad_out = func(x_nonpad)  # [N, D_out]

    D_out = x_nonpad_out.shape[-1]
    x_out = torch.zeros(B, L, D_out, dtype=x_nonpad_out.dtype, device=x.device)
    x_out[~mask] = x_nonpad_out

    return x_out


class MILBase(nn.Module):
    def __init__(self, input_dim, n_classes, dropout, act, mil_norm=None, mil_bias=False, inner_dim=512,
                 aggregate_fn=DAttention, embed_feat=True, embed_norm_pos=0, **aggregate_args):
        super(MILBase, self).__init__()
        self.L = inner_dim  # 512
        self.K = 1
        self.embed = []
        self.mil_norm = mil_norm
        self.embed_norm_pos = embed_norm_pos
        self.input_dim = input_dim
        
        assert self.embed_norm_pos in (0, 1)
        if type(aggregate_fn) == SAttention and mil_norm == 'ln':
            assert embed_norm_pos == 0

        if mil_norm == 'bn':
            self.norm = nn.BatchNorm1d(input_dim) if embed_norm_pos == 0 else nn.BatchNorm1d(inner_dim)
        elif mil_norm == 'ln':
            self.norm = nn.LayerNorm(input_dim, bias=mil_bias) if embed_norm_pos == 0 else nn.LayerNorm(inner_dim,bias=mil_bias)
        else:
            self.norm1 = self.norm = nn.Identity()

        if embed_feat:
            self.embed += [nn.Linear(input_dim, inner_dim, bias=mil_bias)]
            if act.lower() == 'gelu':
                self.embed += [nn.GELU()]
            else:
                self.embed += [nn.ReLU()]

            if dropout:
                self.embed += [nn.Dropout(0.25)]

        self.embed = nn.Sequential(*self.embed) if len(self.embed) > 0 else nn.Identity()

        self.aggregate = aggregate_fn(inner_dim=inner_dim, mil_bias=mil_bias, mil_norm=mil_norm, **aggregate_args)

        if n_classes == 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(self.L * self.K, n_classes, bias=mil_bias)

    def forward_norm(self, x, pack_args, ban_bn=False):
        if self.mil_norm == 'bn' and not ban_bn:
            if pack_args is not None:
                key_pad_mask = pack_args['key_pad_mask_no_cls']
                if pack_args['no_norm_pad']:
                    x = apply_function_nonpad(x, key_pad_mask, self.norm)
                else:
                    x = torch.transpose(x, -1, -2)
                    x = self.norm(x)
                    x = torch.transpose(x, -1, -2)
            else:
                x = torch.transpose(x, -1, -2)
                x = self.norm(x)
                x = torch.transpose(x, -1, -2)
        else:
            if pack_args is not None:
                key_pad_mask = pack_args['key_pad_mask_no_cls']
                if pack_args['no_norm_pad']:
                    x = apply_function_nonpad(x, key_pad_mask, self.norm)
                else:
                    x = self.norm(x)
            else:
                x = self.norm(x)

        return x

    def forward(self, x, pack_args=None, return_attn=False, return_act=False, ban_norm=False, ban_embed=False, residual=False, **mil_kwargs):
        if len(x.size()) == 2:
            x.unsqueeze_(0)

        if not ban_embed:
            if self.embed_norm_pos == 0 and not ban_norm:
                x = self.forward_norm(x, pack_args)

            if pack_args is not None:
                x = apply_function_nonpad(x, pack_args['key_pad_mask_no_cls'], self.embed)
            else:
                x = self.embed(x)

            if self.embed_norm_pos == 1 and not ban_norm:
                x = self.forward_norm(x, pack_args)

        if return_act:
            act = x.clone()

        if return_attn:
            x, attn = self.aggregate(x, pack_args=pack_args, return_attn=return_attn, ban_norm=ban_norm, **mil_kwargs)
        else:
            x = self.aggregate(x, pack_args=pack_args,ban_norm=ban_norm,residual=residual, **mil_kwargs)

        if not isinstance(self.aggregate, DSAttention):
            x = self.classifier(x)

        if return_attn:
            output = []
            output.append(x)
            output.append(attn.squeeze(1))
            if return_act:
                output.append(act.squeeze(1))
            return output
        else:
            return x