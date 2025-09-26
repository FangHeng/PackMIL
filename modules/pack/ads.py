import torch
from torch import Tensor, nn
import torch.nn.functional as F

class DimensionDrop(torch.nn.Module):
    def __init__(self, p=0.5, dim=-1, per_instance=False, scale=False):
        super().__init__()
        self.p = p
        self.dim = dim
        self.scaling = 1 / (1 - p) if scale else 1.
        self.per_instance = per_instance

    def forward(self, x, disable=False):
        if not self.training or self.p == 0 or disable:
            return x

        dim_size = x.size(self.dim)
        keep = int(round(dim_size * (1 - self.p)))

        if self.per_instance:
            noise = torch.rand_like(x, device=x.device)
            shuffle_idx = torch.argsort(noise, dim=self.dim)
            keep_idx = shuffle_idx.narrow(self.dim, 0, keep)
            keep_idx, _ = torch.sort(keep_idx, dim=self.dim)
            x = x.gather(self.dim, keep_idx) * self.scaling
        else:
            noise = torch.rand(x.shape[self.dim], device=x.device)
            shuffle_idx = torch.argsort(noise, dim=0)
            keep_idx = shuffle_idx[:keep].sort().values
            x = x.index_select(self.dim, keep_idx) * self.scaling
        return x


class ADS(nn.Module):
    """
    Downsample the input 3D tensor (B, T, D):
        1) First calculate the attention weight and weight the features
        2) Use shortcut connections to retain the original information
        3) Then rearrange pixels and transform features according to the downsampling factor r
    """

    def __init__(self, r: int = 1, D: int = 1024, act: str = 'relu', hidden_dim: int = None,
                 include_padding: bool = True, attn_dim: int = 128, downsample: int = None,
                 downsample_type: str = 'random', norm: bool = False, force_mlp: bool = False,
                 _type: str = 'random', layer_num=2, attn_act: str = 'tanh', bias=True, no_para_downsample=False,
                 downsample_scale=False, no_pad=False):
        super(ADS, self).__init__()
        self.r = r
        self.D = D
        self.include_padding = True
        self.force_mlp = force_mlp
        self.type = _type
        self.pool_factor = None
        self.no_pad = no_pad
        if self.r > 3 and (downsample or 0) > 1:
            assert self.r % downsample == 0
            self.pool_factor = downsample

        self.pool_type = downsample_type
        self.no_para_downsample = no_para_downsample

        if (r <= 1 or (self.pool_factor and self.r // self.pool_factor == 1)) and not force_mlp:
            self.downsample = None
            if no_para_downsample:
                self.attention = nn.Sequential(
                    nn.Linear(D, attn_dim),
                    nn.Tanh() if attn_act.lower() == 'tanh' else nn.GELU(),
                    nn.Linear(attn_dim, 1)
                )
        else:
            # attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(D, attn_dim),
                nn.Tanh() if attn_act.lower() == 'tanh' else nn.GELU(),
                nn.Linear(attn_dim, 1)
            )

            if hidden_dim is None or layer_num == 1:
                hidden_dim = D
            if hidden_dim < 10:
                hidden_dim = D * hidden_dim
            hidden_dim = int(hidden_dim)

            if self.pool_factor is not None:
                self.in_features = D
            else:
                self.in_features = D
            self.out_features = D
            self.downsample_scale = downsample_scale
            pool_factor = self.pool_factor or 1
            self.drop = DimensionDrop(1 - 1 / (r // pool_factor), per_instance=self.type == 'random_pi',
                                      scale=downsample_scale)

            # construct the MLP for downsampling: (D*r)->D
            if layer_num > 1 and not no_para_downsample:
                self.embed = nn.Sequential(
                    nn.Linear(self.in_features, hidden_dim, bias=bias),
                    nn.GELU() if act.lower() == 'gelu' else nn.ReLU(),
                )
                self.downsample = nn.Linear(hidden_dim, self.out_features, bias=bias)
            elif not no_para_downsample:
                self.embed = nn.Sequential(
                    nn.Linear(self.in_features, hidden_dim, bias=bias),
                    nn.GELU() if act.lower() == 'gelu' else nn.ReLU(),
                )
                self.downsample = nn.Identity()
            else:
                self.embed = nn.Identity()
                self.downsample = nn.Identity()

        self.norm = nn.LayerNorm(self.out_features) if norm else nn.Identity()

    def forward(self, x: torch.Tensor, shuffle: bool = True, key_pad_mask: torch.Tensor = None, downsample=True,
                downsample_pool=False):
        if self.downsample is None:
            if self.no_para_downsample:
                attn = self.attention(x)  # (B, T, 1)
                attn_weights = F.softmax(attn, dim=1)  # (B, T, 1)

                x_attn = x * attn_weights
                x = x + x_attn  # shortcut

            if self.pool_factor is not None:
                B, T, D = x.shape
                if self.pool_type == 'random':
                    indices = torch.randperm(T, device=x.device)[:T // self.pool_factor]
                    x = x[:, indices, :]
                else:
                    if shuffle:
                        perm = torch.randperm(T, device=x.device)
                        x = x[:, perm, :]

                    x = x.view(B, T // self.r, self.r, D)  # (B, T//r, r, D)

                    if self.pool_type == 'mean':
                        x = x.mean(dim=2)  # (B, T//r, D, _r)
                    elif self.pool_type == 'max':
                        x, _ = x.max(dim=2)
                    x = x.view(B, T // self.r, D)  # (B, T//r, D*r//pool_factor)

            return x, None

        if not downsample_pool:
            downsample_pool = downsample if downsample else False

        B, T, D = x.shape
        assert D == self.D, (
            f"Incorrect input dimension D={D}, expected D={self.D}."
        )

        attn = self.attention(x)  # (B, T, 1)
        attn_weights = F.softmax(attn, dim=1)  # (B, T, 1)

        x_attn = x * attn_weights
        x = x + x_attn  # shortcut

        if T % self.r != 0 and self.training and downsample:
            T_pad = (T + self.r - 1) // self.r * self.r
            x_pad = x.new_zeros((B, T_pad, D))
            x_pad[:, :T, :] = x
            x = x_pad
            T = T_pad

        if shuffle:
            perm = torch.randperm(T, device=x.device)
            x = x[:, perm, :]

        r = self.r
        if self.pool_factor is not None and self.pool_type == 'random' and downsample_pool:
            indices = torch.randperm(T, device=x.device)[:T // self.pool_factor]
            x = x[:, indices, :]
            r = self.r // self.pool_factor
            T = T // self.pool_factor

        # Step 1: embed the input features
        # (B, T, D) -> (B, T//r, r, D) -> (B, T//r, D, r) -> (B, T//r, D*r)
        x = self.embed(x)
        if self.training and downsample:
            _D = x.size(-1)
            x = x.view(B, T // r, r, _D)  # (B, T//r, r, D)
            if self.type == 'max':
                x, _ = torch.max(x, dim=-2)
                if self.downsample_scale:
                    x = x * r
            elif self.type == 'mean':
                x = x.mean(dim=-2)
            elif 'random' in self.type:
                x = x.permute(0, 1, 3, 2).contiguous()  # (B, T//r, D, r)
                x = x.view(B, T // r, _D * r)  # (B, T//r, D*r)
                x = self.drop(x, disable=not downsample)

        if self.pool_factor is not None and self.pool_type != 'random':
            raise NotImplementedError

        # Step 2: downsample the features (D*r) -> D
        x = self.downsample(x)  # (B*(T//r), D)

        if downsample and self.training:
            # Step 3: reshape the output to (B, T//r, D)
            x = x.view(B, T // r, self.out_features)

        return self.norm(x), None