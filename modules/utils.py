import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from einops import repeat,rearrange
import math
from copy import deepcopy
import collections.abc
import itertools
from functools import reduce
from operator import mul
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List,Literal
try:
    from flash_attn.flash_attn_interface import _get_block_size_n
except:
    pass

#helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    return lambda *args: val

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(itertools.repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class MCA(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()

        dim_head = dim // heads

        inner_dim = dim_head *  heads
        #project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        #self.scale = dim_head ** -0.5

        #self.attend = nn.Softmax(dim = -1)
        #self.dropout = nn.Dropout(dropout)

        self.to_kv = NoInitLinear(dim, inner_dim * 2, bias = False)
        self.to_q = NoInitLinear(dim, inner_dim, bias = False)

        self.proj = NoInitLinear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.to_kv.requires_grad_(False)
        self.to_q.requires_grad_(False)
        self.proj.requires_grad_(False)

    def forward(self, x, _q, weights = None):
        if weights is not None:
            self.to_q.weight = weights['qkv'].chunk(3, dim = -1)[0]
            self.to_kv.weight = weights['qkv'].chunk(3, dim = -1)[1:]
            self.proj.weight = weights['proj'][0]
            self.proj.bias = weights['proj'][1]

        kv = self.to_kv(x).chunk(2, dim = -1)
        q = self.to_q(_q)

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                    q, k, v,
                )
        
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)

        return out

class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """
    return_indices: torch.jit.Final[bool]

    def __init__(
            self,
            prob: float = 0.5,
            num_prefix_tokens: int = 1,
            ordered: bool = False,
            return_indices: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered
        self.return_indices = return_indices

    def forward(self, x ,num_prefix_tokens=None,**kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if not self.training or self.prob == 0.:
            if self.return_indices:
                return x, None
            return x

        _num_prefix_tokens = num_prefix_tokens or self.num_prefix_tokens

        if _num_prefix_tokens:
            prefix_tokens, x = x[:, :_num_prefix_tokens], x[:, _num_prefix_tokens:]
        else:
            prefix_tokens = None

        B = x.shape[0]
        L = x.shape[1]
        num_keep = max(1, int(L * (1. - self.prob)))
        keep_indices = torch.argsort(torch.randn(B, L, device=x.device), dim=-1)[:, :num_keep]
        if self.ordered:
            # NOTE does not need to maintain patch order in typical transformer use,
            # but possibly useful for debug / visualization
            keep_indices = keep_indices.sort(dim=-1)[0]
        x = x.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))

        if prefix_tokens is not None:
            x = torch.cat((prefix_tokens, x), dim=1)

        if self.return_indices:
            return x, keep_indices
        return x

class PatchMerge(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0., k=10,merge_ratio=0.2, mask_type='random',deep_depth=1,num_prefix_tokens: int = 1,return_indices: bool = False,):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MCA(dim,heads)
        self.prompt_dropout = nn.Dropout(dropout)
        self.merge_k = k 
        self.deep_depth = deep_depth
        self.mask_type = mask_type
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.return_indices = return_indices

        assert deep_depth > 0

        self.global_q_grad = nn.Parameter(torch.zeros(deep_depth, k, dim),requires_grad=True)
        # copy from vpt@google
        val = math.sqrt(6. / float(3 * reduce(mul, (14,14), 1) + dim))  # noqa
        nn.init.uniform_(self.global_q_grad.data, -val, val)
        self.global_q = self.global_q_grad

        self.merge_ratio = merge_ratio
        self.k = k

    def merge(self,x,depth=0):
        B = x.shape[0]
        depth = min(depth,self.global_q.shape[0]-1,0)
        global_q = self.prompt_dropout(self.global_q[depth]).unsqueeze(0).expand(B, -1, -1)
        return self.attn(self.norm(x),self.norm(global_q))
    
    def masking(self,x,attn,num_prefix_tokens=None):
        B,L,C = x.shape

        _num_prefix_tokens = num_prefix_tokens or self.num_prefix_tokens

        if _num_prefix_tokens:
            prefix_tokens, x = x[:, :_num_prefix_tokens], x[:, _num_prefix_tokens:]
        else:
            prefix_tokens = None

        B = x.shape[0]
        L = x.shape[1]
        merge_ratio = self.merge_ratio
        
        # Calculate indices for kept and dropped tokens
        if self.mask_type == 'random':
            noise = torch.rand(B, L, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=-1)
            len_keep = int(L * (1-merge_ratio))
            ids_keep = ids_shuffle[:, :len_keep]
            ids_drop = ids_shuffle[:, len_keep:]
        elif self.mask_type == 'low':
            raise NotImplementedError
        
        # Get kept and dropped tokens
        x_keep = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, C))
        x = x.gather(1, ids_drop.unsqueeze(-1).expand(-1, -1, C))

        if prefix_tokens is not None:
            x_keep = torch.cat((prefix_tokens, x_keep), dim=1)

        return x_keep, x, ids_keep

    def forward(self,x,attn=None,num_prefix_tokens=None,depth=0):
        if self.training and self.merge_ratio > 0:
            x_keep, x, ids_keep = self.masking(x,attn,num_prefix_tokens=num_prefix_tokens)
            # if self.no_merge:
            #     if self.global_q is not None:
            #         x_keep = torch.cat((x_keep,self.global_q),dim=1)
            #     return x_keep
            # else:
            x = torch.cat((self.merge(x,depth=depth),x_keep),dim=1)
            if self.return_indices:
                return x,ids_keep
            return x
        else:
            return x
            # if not self.no_merge:
            #     return torch.cat((x,self.merge(x)),dim=1) 
            # else:
            #     if self.global_q is not None:
            #         x = torch.cat((x,self.global_q),dim=1)
            #     return x

class NoInitLinear(nn.Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.parameter.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

def check_tensor(tensor, tensor_name=""):
    if torch.isnan(tensor).any():
        print(f"{tensor_name} contains NaN values")
    if torch.isinf(tensor).any():
        print(f"{tensor_name} contains Inf values")
    if torch.isfinite(tensor).all():
        print(f"{tensor_name} normal ")

def get_mil_model_params_from_name(args,name):
    model_params = {}
    if name == 'abmil':
        model_params.update({
            "da_gated": args.da_gated
        })
    elif name == 'dsmil':
        model_params.update({
            "ds_average": args.ds_average
        })
    
    return model_params

def get_mil_model_params(args):
    genera_model_params = {
        "input_dim": args.input_dim,
        "n_classes": args.n_classes,
        "dropout": args.dropout,
        "act": args.act,
        "mil_norm": args.mil_norm,
        "cls_norm": args.cls_norm,
        "mil_bias": args.mil_bias,
        "inner_dim": args.inner_dim,
        "embed_feat": args.mil_feat_embed,
        'embed_feat_mlp_ratio': args.mil_feat_embed_mlp_ratio,
        'fc_norm_bn': True,
        'embed_norm_pos': args.embed_norm_pos,
        'feat_embed_type': args.mil_feat_embed_type,
        'pos': args.pos,
    }
    genera_trans_params = deepcopy(genera_model_params)
    genera_trans_params.update({
        'n_layers': args.n_layers,
        'pool': args.pool,
        'attn_dropout': args.attn_dropout,
        'deterministic': not args.no_determ,
        'ffn': args.ffn,
        'sdpa_type': args.sdpa_type,
        'n_heads':args.n_heads,
        'fc_norm':True,
        'vit_norm': True,
        'attn_type': args.attn_type,
        'ffn_bias': True,
        'ffn_dp': 0.,
        'ffn_ratio': 4.,
    })

    return genera_model_params,genera_trans_params

def soft_ce(x: torch.Tensor, target: torch.Tensor, mean: bool= True, temp_t: float=1., temp_s: float=1.):
    loss = torch.sum(-F.softmax(target / temp_t,dim=-1) * F.log_softmax(x / temp_s, dim=-1), dim=-1)
    if mean:
        return loss.mean()
    else:
        return loss

#### flash attn
def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )

def convert_flash_attn_S_to_softmax(
    S,
    seqlen_q,
    seqlen_k,
    query_padding_mask,
    key_padding_mask,
    head_dim,
    is_dropout,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
        query_padding_mask: (batch_size, seqlen_q_rounded)
        key_padding_mask: (batch_size, seqlen_k_rounded)
    """
    if causal:
        window_size = (window_size[0], 0)
    seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
    S_converted = S
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            S.device,
        )
        local_mask = F.pad(
            local_mask,
            (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
            value=True,
        )
        S_converted = S_converted.masked_fill(local_mask, 0.0)

    # Need to zero out things not in attention_mask in case S was initialized with random values
    # and some of those values aren't overwritten.
    seqlen_q_og = (
        query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
    )
    if query_padding_mask is not None:
        query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
        S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
    if key_padding_mask is not None:
        key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
        S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
    S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
    S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
    return S_converted[:, :, :seqlen_q, :seqlen_k]

def normalize_flash_attn_S(
    attn_unnorm,
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    is_dropout=False,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k, v: (batch_size, seqlen_k, nheads, head_dim)
        key_padding_mask: (batch_size, seqlen_q)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
    Output:
        softmax_lse: (batch_size, nheads, seqlen_q)
        softmax_max: (batch_size, nheads, seqlen_q)
    """
    if causal:
        window_size = (window_size[0], 0)
    q, k, v = q.float(), k.float(), v.float()
    _, seqlen_q, _, head_dim = q.shape
    seqlen_k = k.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(head_dim), k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias.to(dtype=scores.dtype)
    block_size_n = _get_block_size_n(scores.device, head_dim, is_dropout, causal)
    scores_block = scores.split(block_size_n, dim=-1)
    lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
    lse = torch.logsumexp(lse_block, dim=-1)
    # lse could be -inf (i.e. all values in scores are -inf), and we want to set those to inf
    # so that when we do torch.exp(m - lse), we get 0.0 instead of NaN.
    lse[lse == float("-inf")] = float("inf")
    scores_max_block = torch.stack([torch.amax(s, dim=-1) for s in scores_block], dim=-1)
    cummax_block = torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
    attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
    attn_norm = torch.cat(
        [
            a * rearrange(torch.exp(m - lse), "b h s -> b h s 1")
            for a, m in zip(attn_unnorm_block, cummax_block)
        ],
        dim=-1,
    )
    if query_padding_mask is not None:
        attn_norm.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    return attn_norm.to(dtype=attn_unnorm.dtype)