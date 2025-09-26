import torch
from torch import nn
import numpy as np
from timm.loss import AsymmetricLossSingleLabel

from .pack.pack_baseline import MILBase, DAttention, SAttention, DSAttention, RRTMIL
from .pack.pack_util import *
from .pack.packing import get_packs
from .pack.ads import ADS
from .pack.pack_loss import NLLSurvMulLoss, BCESurvLoss, FocalLoss

class PackMIL(nn.Module):
    def __init__(self, mil='abmil', task_type="sub", token_dropout=0.5, group_max_seq_len=2048, min_seq_len=512,
                 pack_residual=True, downsample_mode='none', residual_loss='bce', residual_downsample_r=1,
                 residual_ps_weight=False, pad_r=False, singlelabel=False, residual_type='norm',
                 downsample_type='random', **mil_kwargs):
        super(PackMIL, self).__init__()

        if 'mil_norm' in mil_kwargs:
            if mil_kwargs['mil_norm'] == 'bn':
                no_norm_pad = True
        self.pool = mil_kwargs.get('pool', 'cls_token')
        mil_kwargs['attn_type'] = 'naive'
        if self.pool == 'cls_token':
            mil_kwargs['pool'] = 'cls_token'
        self.need_attn_mask = False
        self.n_classes = mil_kwargs.pop('n_classes')
        mil_kwargs['n_classes'] = 0

        self.task_type = task_type  # surv, grade, subtype
        self.downsample_mode = downsample_mode

        if mil == 'abmil':
            self.mil = MILBase(aggregate_fn=DAttention, **mil_kwargs)
            self.pool = None
        elif mil == 'dsmil':
            _n_classes = mil_kwargs.pop('n_classes')
            mil_kwargs['agg_n_classes'] = _n_classes
            self.mil = MILBase(aggregate_fn=DSAttention, **mil_kwargs)
            self.pool = None
        elif mil == 'vitmil':
            self.mil = MILBase(aggregate_fn=SAttention, **mil_kwargs)
            self.need_attn_mask = True
        elif mil == 'transmil':
            mil_kwargs['attn_type'] = 'ntrans'
            mil_kwargs['pos'] = 'ppeg'
            self.mil = MILBase(aggregate_fn=SAttention, **mil_kwargs)
        elif mil == 'rrtmil':
            self.mil = MILBase(aggregate_fn=RRTMIL, **mil_kwargs)
            self.pool = None
        else:
            raise NotImplementedError

        self.predictor = nn.Linear(mil_kwargs['inner_dim'], self.n_classes) if self.n_classes > 0 else nn.Identity()
        
        if pack_residual:
            if residual_loss == 'bce':
                self.residual_loss = nn.BCEWithLogitsLoss()
            elif residual_loss == 'asl_single':
                self.residual_loss = AsymmetricLossSingleLabel(gamma_pos=1, gamma_neg=4, eps=0.)
            elif residual_loss == 'focal':
                self.residual_loss = FocalLoss(alpha=0.25, gamma=2.0)
            elif residual_loss == 'ce':
                self.residual_loss = nn.CrossEntropyLoss()
            elif residual_loss == 'nll':
                self.residual_loss = NLLSurvMulLoss()

            _n_classes_mix = self.n_classes if self.n_classes > 0 else _n_classes
            if mil == 'dsmil':
                if residual_type == 'dual_cls':
                    self.predictor_res = nn.Conv1d(_n_classes_mix, _n_classes_mix, kernel_size=mil_kwargs['inner_dim'])
                else:
                    self.predictor_res = None
            else:
                self.predictor_res = nn.Linear(mil_kwargs['inner_dim'], _n_classes_mix) if residual_type == 'dual_cls' else None
        else:
            self.residual_loss = None
            self.predictor_res = None

        self.token_dropout = token_dropout
        self.group_max_seq_len = group_max_seq_len
        self.min_seq_len = min_seq_len
        self.no_norm_pad = no_norm_pad
        self.residual = pack_residual
        self.residual_ps_weight = residual_ps_weight
        self.downsample_r = residual_downsample_r

        if self.downsample_mode == 'ads':
            self.downsampler = ADS(r=self.downsample_r,
                                   D=mil_kwargs.get('input_dim', 1024),
                                   _type=downsample_type
                                   )
            self.all_ds = True
        else:
            self.downsampler = None
            self.all_ds = False

        self.pad_r = pad_r
        self.singlelabel = singlelabel

        self.apply(initialize_weights)

    def apply_inference_downsample(self, x):
        if self.downsampler is not None:
            _tmp = self.downsampler.pool_factor
            _tmp_r = self.downsampler.r
            self.downsampler.pool_factor = None
            if _tmp is not None:
                self.downsampler.r = _tmp_r // _tmp
            x = self.downsampler(x, shuffle=False)
            self.downsampler.pool_factor = _tmp
            self.downsampler.r = _tmp_r

        return x

    def forward(self, x, label=None, loss=None, pos=None, **mil_kwargs):
        _pn = sum([len(_x) for _x in x])
        if self.training:
            B = len(x)

            if self.token_dropout > 0:
                _token_dropout = self.token_dropout

                max_feat_num = max([feat.size(0) for feat in x])
                keep_rate = 1 - self.token_dropout
                max_feat_num = int(int(max_feat_num * keep_rate) / self.downsample_r)
                if max_feat_num > self.group_max_seq_len:
                    pack_len = self.group_max_seq_len * 2
                else:
                    pack_len = self.group_max_seq_len

                if self.task_type == 'surv':
                    y, c = label
                    _label = torch.cat([y.unsqueeze(-1), c.unsqueeze(-1)], dim=1)
                else:
                    _label = label

                kept_dict, drop_dict = get_packs(
                    x=x,
                    token_dropout=_token_dropout,
                    group_max_seq_len=pack_len,
                    min_seq_len=self.min_seq_len,
                    pool=self.pool,
                    device=x[0].device,
                    need_attn_mask=self.need_attn_mask,
                    labels=_label,
                    poses=pos,
                    residual=self.residual,
                    seq_downsampler=self.downsampler,
                    enable_drop=self.residual,
                    all_pu=self.all_ds,
                    pad_r=self.pad_r,
                )
                x, attn_mask, key_pad_mask, num_feats, batched_feat_ids, cls_token_mask, batched_feat_ids_1, key_pad_mask_no_cls, pos, batched_label, batched_num_ps = kept_dict.values()

                pack_args = {
                    'attn_mask': attn_mask,
                    'num_images': num_feats,
                    'batched_image_ids': batched_feat_ids,
                    'batched_image_ids_1': batched_feat_ids_1,
                    'key_pad_mask': key_pad_mask,
                    'key_pad_mask_no_cls': key_pad_mask_no_cls,
                    'cls_token_mask': cls_token_mask,
                    'no_norm_pad': self.no_norm_pad,
                    'residual': False,
                }

                _kn = torch.sum(~key_pad_mask).item()
                total_elements = torch.numel(key_pad_mask)
                _pr = (1 - (_kn / total_elements)) * 100
                flattened_num_ps = [item for sublist in batched_num_ps for item in sublist]
                _kn_std = np.std(np.array(flattened_num_ps))
            else:
                x = torch.cat(x, dim=0).view(B, -1, self.mil.input_dim)
                pack_args = None
                _kn = _pn
                _pr = 0.
                _kn_std = 0.

            if self.residual:
                x_res = drop_dict['patches']
                label_res = drop_dict['batched_labels']
                key_pad_mask = drop_dict['key_pad_mask']
                key_pad_mask_res = drop_dict['key_pad_mask_res']
                key_pad_mask_no_cls_res = drop_dict['key_pad_mask_no_cls']
                batched_num_ps_res = drop_dict['batched_num_ps']

                if self.need_attn_mask:
                    global_mask = ~key_pad_mask_res.cpu()
                    global_attn_mask = (global_mask.unsqueeze(2) & global_mask.unsqueeze(1)).unsqueeze(1)
                    global_attn_mask = global_attn_mask.to(x.device)
                else:
                    global_attn_mask = None

                pack_res_args = {
                    'attn_mask': global_attn_mask,
                    'num_images': None,
                    'batched_image_ids': None,
                    'batched_image_ids_1': None,
                    'key_pad_mask': key_pad_mask_res,
                    'key_pad_mask_no_cls': key_pad_mask_no_cls_res,
                    'cls_token_mask': None,
                    'no_norm_pad': self.no_norm_pad,
                    'residual': True,
                }

                x_res = self.mil(x_res, pack_args=pack_res_args, ban_norm=True, **mil_kwargs)

                if self.predictor_res is not None:
                    # DSMIL
                    if isinstance(x_res, list):
                        _logits_res = self.predictor_res(x_res[-1]).squeeze(-1)
                    else:
                        _logits_res = self.predictor_res(x_res)
                else:
                    _logits_res = self.predictor(x_res)

                if self.task_type == 'surv':
                    _is_multi_lalbel = isinstance(self.residual_loss, BCESurvLoss)
                else:
                    _is_multi_lalbel = not (isinstance(self.residual_loss, nn.CrossEntropyLoss)
                                            or isinstance(self.residual_loss, AsymmetricLossSingleLabel))
                y_res = mixup_target_batched(
                    label_res,
                    num_classes=self.n_classes,
                    multi_label=_is_multi_lalbel,
                    batched_num_ps=batched_num_ps_res if self.residual_ps_weight else None,
                    target_task=self.task_type
                )
                if isinstance(_logits_res, list):
                    if self.task_type == 'surv':
                        aux_loss = self.residual_loss(Y=y_res['Y'], c=y_res['c'], Y_censored=y_res['Y_censored'],
                                                      logits=_logits_res[0])
                    else:
                        if isinstance(self.residual_loss, AsymmetricLossSingleLabel) or self.singlelabel:
                            y_res = y_res.argmax(dim=1)
                        aux_loss = self.residual_loss(_logits_res[0], y_res)

                else:
                    if self.task_type == 'surv':
                        aux_loss = self.residual_loss(Y=y_res['Y'], c=y_res['c'], Y_censored=y_res['Y_censored'],
                                                      logits=_logits_res)

                    else:
                        if isinstance(self.residual_loss, AsymmetricLossSingleLabel) or self.singlelabel:
                            y_res = y_res.argmax(dim=1)
                        aux_loss = self.residual_loss(_logits_res, y_res)

            else:
                aux_loss = 0.

            _logits = self.mil(x, pack_args=pack_args, pos=pos, **mil_kwargs)
            if isinstance(_logits, list):
                if isinstance(loss, nn.CrossEntropyLoss):
                    _loss = loss(_logits[1].view(B, -1), label)
                elif isinstance(loss, nn.BCEWithLogitsLoss):
                    _loss = loss(_logits[1].view(B, -1), label.view(B, -1).float())
                else:
                    _loss = loss(logits=_logits[1].view(B, -1), Y=label[0], c=label[1])
                _logits = [self.predictor(_logits[0]), _loss, None]
            else:
                _logits = self.predictor(_logits)

            return _logits, aux_loss, _pn / B, _kn / B, _pr, _kn_std
        else:
            x_res = self.apply_inference_downsample(x)
            x_res = self.mil(x_res, residual=True, **mil_kwargs)
            if isinstance(x_res, list) and self.predictor_res is not None:
                x_res = [x_res[0], x_res[1]]
            else:
                x_res = self.predictor(x_res)
                if isinstance(x_res, list) and len(x_res) > 2:
                    x_res = [x_res[0], x_res[1]]

            pack_args = None

            if isinstance(x_res, list):
                x[0] = x_res[0]
                x[1] = x_res[1]
            else:
                x = x_res