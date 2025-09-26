from functools import partial
import torch
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence
from modules.pack.pack_util import get_dropout, group_seqs
from modules.pack.ads import ADS
from einops import rearrange, repeat
import torch.nn.functional as F

def get_packs(x,
              token_dropout,
              group_max_seq_len,
              min_seq_len,
              pool,
              device,
              need_attn_mask=True,
              token_dropout_sub=0.,
              poses=None,
              labels=None,
              residual=False,
              grouping_strategy="sequential",
              seq_downsampler=None,
              enable_drop=False,
              all_pu=False,
              pad_r=False,
              all_dual=False):
    """
    Pack the input feature x (shape: [B, N, D]) into multiple packs, and pack the kept and dropped parts separately.
    The dropped part consists of the unselected tokens in the kept part, and is grouped after downsampling.
    """
    arange = partial(torch.arange, device=device)
    pad_sequence = partial(orig_pad_sequence, batch_first=True)

    # [64,N,D] ---> [5,N*,D]

    _x_single = []
    _x_multi = []
    _r = seq_downsampler.r if seq_downsampler is not None else 1
    if seq_downsampler is not None:
        _r = seq_downsampler.r
        _r_pool = seq_downsampler.pool_factor if seq_downsampler.pool_factor is not None else 1
    else:
        _r_pool = 1
    max_pre_len = group_max_seq_len * _r
    for i, feat in enumerate(x):
        # feat: [n, d]
        numpatch = feat.shape[0]
        target_tokens = min_seq_len / token_dropout if all_dual else min_seq_len * _r

        if numpatch < target_tokens:
            feat_single = feat
            feat_multi = feat
        else:
            if all_dual:
                threshold_single = min_seq_len / token_dropout
                threshold_multi = min_seq_len / (1 - token_dropout)
            else:
                threshold_single = target_tokens / token_dropout
                threshold_multi = target_tokens / (1 - token_dropout)
            threshold = max(threshold_single, threshold_multi)

            perm = torch.randperm(numpatch, device=device)

            if numpatch < threshold and pad_r:
                n_allocated_single = int(numpatch * token_dropout)
                n_allocated_multi = numpatch - n_allocated_single

                init_single = perm[:n_allocated_single]
                init_multi = perm[n_allocated_single:]

                if init_single.shape[0] < target_tokens:
                    needed = target_tokens - init_single.shape[0]
                    extra = init_multi[:needed] if init_multi.shape[0] >= needed else init_multi
                    idx_single = torch.cat([init_single, extra])
                else:
                    idx_single = init_single

                if init_multi.shape[0] < target_tokens:
                    needed = target_tokens - init_multi.shape[0]
                    extra = init_single[:needed] if init_single.shape[0] >= needed else init_single
                    idx_multi = torch.cat([init_multi, extra])
                else:
                    idx_multi = init_multi

                feat_single = feat[idx_single, :]
                feat_multi = feat[idx_multi, :]
            else:
                # if numpatch >= threshold, we allocate tokens based on the dropout rate
                n_allocated_single = int(numpatch * token_dropout)
                idx_single = perm[:n_allocated_single]
                idx_multi = perm[n_allocated_single:]
                feat_single = feat[idx_single, :]  # [n_single, d]
                feat_multi = feat[idx_multi, :]  # [n_multi, d]

        if feat_single.shape[0] > max_pre_len:
            rand_indices_1 = torch.randperm(feat_single.shape[0], device=device)[:max_pre_len]
            feat_single = feat_single[rand_indices_1]
        if feat_multi.shape[0] > max_pre_len:
            rand_indices_2 = torch.randperm(feat_multi.shape[0], device=device)[:max_pre_len]
            feat_multi = feat_multi[rand_indices_2]

        if feat_single.numel() > (0 if all_pu else min_seq_len * _r) and seq_downsampler is not None:
            if isinstance(seq_downsampler, ADS):
                feat_single_ds = seq_downsampler(feat_single.unsqueeze(0), shuffle=True,
                                                    downsample=len(feat_single) > min_seq_len * _r,
                                                    )
                feat_multi_ds = seq_downsampler(feat_multi.unsqueeze(0), shuffle=True,
                                                   downsample=len(feat_multi) > min_seq_len * _r,
                                                   )
            else:
                feat_single_ds = seq_downsampler(feat_single.unsqueeze(0), shuffle=True)
                feat_multi_ds = seq_downsampler(feat_multi.unsqueeze(0), shuffle=True)

            feat_single = feat_single_ds.squeeze(0)
            feat_multi = feat_multi_ds.squeeze(0)

        _x_single.append(feat_single)
        _x_multi.append(feat_multi)

    batched_feats_single, batched_num_ps_single, batched_orig_indices_single = group_seqs(
        _x_single,
        poses,
        labels=labels if residual else None,
        token_dropout=token_dropout_sub,
        max_seqs_len=group_max_seq_len,
        min_seq_len=min_seq_len,
        grouping_strategy=grouping_strategy,
        return_indices=True
    )
    dict_single, _ = _process_kept_part(
        batched_feats_single, batched_num_ps_single, batched_orig_indices_single,
        residual, pool, min_seq_len, group_max_seq_len, device,
        need_attn_mask, poses, labels, pad_sequence, orig_pad_sequence
    )

    if enable_drop:
        batched_feats_multi, batched_num_ps_multi, batched_orig_indices_multi = group_seqs(
            _x_multi,
            poses,
            labels=labels if residual else None,
            token_dropout=token_dropout_sub,
            max_seqs_len=group_max_seq_len,
            min_seq_len=min_seq_len,
            grouping_strategy=grouping_strategy,
            return_indices=True
        )
        dict_multi = _process_res_part(
            batched_feats_multi, batched_num_ps_multi, batched_orig_indices_multi,
            residual, pool, min_seq_len, group_max_seq_len, device,
            need_attn_mask, poses, labels, pad_sequence, orig_pad_sequence
        )

        return dict_single, dict_multi
    else:
        return dict_single, None


def _process_kept_part(batched_feats, batched_num_ps, batched_orig_indices,
                       residual, pool, min_seq_len, group_max_seq_len, device,
                       need_attn_mask, poses, labels,
                       pad_sequence, orig_pad_sequence, additional_token_drop=None):
    """
    Process the kept part and group the tokens in each pack in batched_feats in the original way.
    Save the drop token index corresponding to each original sequence in res_indices_dict.
    """
    kept_num_feats = []
    kept_batched_sequences = []
    kept_batched_positions = []
    kept_batched_labels = []
    kept_batched_feat_ids = []
    kept_batched_feat_ids_1 = []
    kept_valid_lengths = []

    res_indices_dict = {}

    for feats, orig_indices in zip(batched_feats, batched_orig_indices):
        kept_sequences = []
        kept_positions = []
        kept_labels = []
        kept_feat_ids = torch.empty((0,), device=device, dtype=torch.long)
        kept_feat_ids_1 = torch.empty((0,), device=device, dtype=torch.long)

        for feat_id, _data in enumerate(feats):
            # _data: [feat, dropout_prob, position] or [feat, dropout_prob]
            if len(_data) == 2:
                feat, dp = _data
                pos = None
            elif len(_data) == 3:
                feat, dp, pos = _data
            if residual:
                feat, label = feat
            else:
                label = None

            orig_idx = orig_indices[feat_id]
            seq = feat
            seq_len = seq.shape[0]

            if dp > 0. and seq_len > min_seq_len:
                num_keep = max(min_seq_len, int(seq_len * (1 - dp)))
                if additional_token_drop is not None:
                    num_keep = min(num_keep, group_max_seq_len * additional_token_drop)
                else:
                    num_keep = min(num_keep, group_max_seq_len)
                scores = torch.randn(seq_len, device=device)
                keep_indices = scores.topk(num_keep, dim=-1).indices
                mask = torch.ones(seq_len, dtype=torch.bool, device=device)
                mask[keep_indices] = False
                drop_indices = mask.nonzero(as_tuple=False).squeeze(-1)

                if drop_indices.numel() < min_seq_len:
                    num_extra = min_seq_len - drop_indices.numel()
                    available = keep_indices
                    # If the number of samples available is sufficient, sampling without replacement is used; otherwise, sampling with replacement is used.
                    if available.numel() >= num_extra:
                        extra_indices = available[torch.randperm(available.numel(), device=device)[:num_extra]]
                    else:
                        extra_indices = available[torch.randint(0, available.numel(), (num_extra,), device=device)]
                    drop_indices = torch.cat([drop_indices, extra_indices])
                res_indices_dict[orig_idx] = drop_indices
                if additional_token_drop is not None:
                    final_num_keep = int(num_keep / additional_token_drop)
                    final_num_keep = max(min_seq_len, final_num_keep)
                    perm = torch.randperm(keep_indices.size(0), device=device)
                    keep_indices = keep_indices[perm[:final_num_keep]]
                kept_seq = seq[keep_indices]
            else:
                kept_seq = seq
                res_indices_dict[orig_idx] = torch.arange(seq_len, device=device)

            token_len = 1 if pool == 'cls_token' else 0

            kept_feat_ids = F.pad(kept_feat_ids, (0, kept_seq.shape[-2] + token_len), value=feat_id)
            kept_feat_ids_1 = torch.cat([torch.zeros((token_len,), device=device),
                                         F.pad(kept_feat_ids_1, (0, kept_seq.shape[-2]), value=feat_id + 1)])
            kept_sequences.append(kept_seq)
            kept_positions.append(pos)
            kept_labels.append(label)

        # [ (n1, d), (n2, d), ..., (nk, d) ] --> [ (n1 + n2 + ... + nk), d ]
        kept_packed_seq = torch.cat(kept_sequences, dim=0) if len(kept_sequences) > 0 else torch.empty(0, device=device)
        if any(p is not None for p in kept_positions):
            kept_packed_pos = torch.cat([p for p in kept_positions if p is not None], dim=0)
        else:
            kept_packed_pos = None
        curr_kept_len = kept_packed_seq.shape[0]

        kept_valid_lengths.append(curr_kept_len)
        kept_batched_sequences.append(kept_packed_seq)
        if kept_packed_pos is not None:
            kept_batched_positions.append(kept_packed_pos)
        if any(l is not None for l in kept_labels):
            kept_batched_labels.append(torch.stack([l for l in kept_labels if l is not None]))
        kept_batched_feat_ids.append(kept_feat_ids)
        kept_batched_feat_ids_1.append(kept_feat_ids_1)
        kept_num_feats.append(len(feats))

    # derive key padding mask for kept part
    if pool == 'cls_token':
        kept_lengths = torch.tensor(
            [l + (n * 1) for l, n in zip(kept_valid_lengths, kept_num_feats)],
            device=device,
            dtype=torch.long
        )
        kept_lengths_no_cls = torch.tensor(kept_valid_lengths, device=device, dtype=torch.long)

        max_length_val_no_cls = kept_lengths_no_cls.amax().item() if len(kept_lengths_no_cls) > 0 else 0
        max_length_no_cls = torch.arange(max_length_val_no_cls, device=device)
        kept_key_pad_mask_no_cls = rearrange(kept_lengths_no_cls, 'b -> b 1') <= rearrange(max_length_no_cls,
                                                                                           'n -> 1 n')
    else:
        kept_lengths = torch.tensor(kept_valid_lengths, device=device, dtype=torch.long)
        kept_key_pad_mask_no_cls = None

    max_length_val = kept_lengths.amax().item() if len(kept_lengths) > 0 else 0
    max_length = torch.arange(max_length_val, device=device)
    kept_key_pad_mask = rearrange(kept_lengths, 'b -> b 1') <= rearrange(max_length, 'n -> 1 n')

    kept_batched_feat_ids = pad_sequence(kept_batched_feat_ids)
    kept_batched_feat_ids_1 = pad_sequence(kept_batched_feat_ids_1)
    if need_attn_mask:
        kept_attn_mask = rearrange(kept_batched_feat_ids, 'b i -> b 1 i 1') == rearrange(kept_batched_feat_ids,
                                                                                         'b j -> b 1 1 j')
        kept_attn_mask = kept_attn_mask & rearrange(~kept_key_pad_mask, 'b j -> b 1 1 j')
    else:
        kept_attn_mask = None

    kept_patches = orig_pad_sequence(kept_batched_sequences, batch_first=True, padding_value=0.)
    if poses is not None:
        kept_patch_positions = pad_sequence(kept_batched_positions)
    else:
        kept_patch_positions = None

    if pool == 'cls_token':
        kept_cls_token_mask = torch.zeros_like(kept_batched_feat_ids, dtype=torch.bool)
        for i in range(kept_batched_feat_ids.shape[0]):
            valid_mask = ~kept_key_pad_mask[i]
            feat_changes = torch.cat([
                torch.tensor([True], device=device),
                (kept_batched_feat_ids[i, 1:] != kept_batched_feat_ids[i, :-1]) & valid_mask[1:]
            ])
            kept_cls_token_mask[i] = feat_changes & valid_mask
    else:
        kept_cls_token_mask = None

    if kept_key_pad_mask_no_cls is None:
        kept_key_pad_mask_no_cls = kept_key_pad_mask

    kept_num_feats_tensor = torch.tensor(kept_num_feats, device=device, dtype=torch.long)

    kept_dict = {
        "patches": kept_patches,
        "attn_mask": kept_attn_mask,
        "key_pad_mask": kept_key_pad_mask,
        "num_feats": kept_num_feats_tensor,
        "batched_feat_ids": kept_batched_feat_ids,
        "cls_token_mask": kept_cls_token_mask if pool == 'cls_token' else None,
        "batched_feat_ids_1": kept_batched_feat_ids_1,
        "key_pad_mask_no_cls": kept_key_pad_mask_no_cls,
        "patch_positions": kept_patch_positions,
        "batched_labels": kept_batched_labels,
        "batched_num_ps": batched_num_ps
    }
    return kept_dict, res_indices_dict


def _process_res_part(batched_feats, batched_num_ps, batched_orig_indices,
                      residual, pool, min_seq_len, group_max_seq_len, device,
                      need_attn_mask, poses, labels,
                      pad_sequence, orig_pad_sequence):
    res_num_feats = []
    res_batched_sequences = []
    res_batched_positions = []
    res_batched_labels = []
    res_batched_feat_ids = []
    res_batched_feat_ids_1 = []
    res_valid_lengths = []

    for feats, orig_indices in zip(batched_feats, batched_orig_indices):
        res_sequences = []
        res_positions = []
        res_labels_pack = []
        res_feat_ids = torch.empty((0,), device=device, dtype=torch.long)
        res_feat_ids_1 = torch.empty((0,), device=device, dtype=torch.long)

        for feat_id, _data in enumerate(feats):
            if len(_data) == 2:
                feat, _ = _data
                pos = None
            elif len(_data) == 3:
                feat, _, pos = _data
            if residual:
                feat, label = feat
            else:
                label = None

            token_len = 1 if pool == 'cls_token' else 0
            orig_idx = orig_indices[feat_id]
            res_feat_ids = F.pad(res_feat_ids, (0, feat.shape[-2] + token_len), value=orig_idx)
            if need_attn_mask:
                res_feat_ids_1 = torch.cat([
                    torch.zeros((token_len,), device=device),
                    F.pad(res_feat_ids_1, (0, feat.shape[-2]), value=orig_idx + 1)
                ])
            res_sequences.append(feat)
            res_positions.append(pos)
            res_labels_pack.append(label)

        res_packed_seq = torch.cat(res_sequences, dim=0) if len(res_sequences) > 0 else torch.empty(0, device=device)
        if any(p is not None for p in res_positions):
            res_packed_pos = torch.cat([p for p in res_positions if p is not None], dim=0)
        else:
            res_packed_pos = None

        curr_res_len = res_packed_seq.shape[0]
        res_valid_lengths.append(curr_res_len)
        res_batched_sequences.append(res_packed_seq)
        if res_packed_pos is not None:
            res_batched_positions.append(res_packed_pos)
        if any(l is not None for l in res_labels_pack):
            res_batched_labels.append(torch.stack([l for l in res_labels_pack if l is not None]))
        res_batched_feat_ids.append(res_feat_ids)
        if need_attn_mask:
            res_batched_feat_ids_1.append(res_feat_ids_1)
        res_num_feats.append(len(feats))

    if pool == 'cls_token':
        res_lengths = torch.tensor(
            [l + (n * 1) for l, n in zip(res_valid_lengths, res_num_feats)],
            device=device,
            dtype=torch.long
        )
        res_lengths_no_cls = torch.tensor(res_valid_lengths, device=device, dtype=torch.long)

        res_max_length_val_no_cls = res_lengths_no_cls.amax().item() if len(res_lengths_no_cls) > 0 else 0
        max_length_no_cls = torch.arange(res_max_length_val_no_cls, device=device)
        res_key_pad_mask_no_cls = rearrange(res_lengths_no_cls, 'b -> b 1') <= rearrange(max_length_no_cls, 'n -> 1 n')
    else:
        res_lengths = torch.tensor(res_valid_lengths, device=device, dtype=torch.long)
        res_key_pad_mask_no_cls = None

    res_max_length_val = res_lengths.amax().item() if len(res_lengths) > 0 else 0
    max_length = torch.arange(res_max_length_val, device=device)
    res_key_pad_mask = rearrange(res_lengths, 'b -> b 1') <= rearrange(max_length, 'n -> 1 n')

    if pool == 'cls_token':
        res_lengths_res = torch.tensor(
            [l + 1 for l in res_valid_lengths],
            device=device,
            dtype=torch.long
        )
    else:
        res_lengths_res = torch.tensor(res_valid_lengths, device=device, dtype=torch.long)

    res_max_length_val_res = (res_lengths_res.amax().item() if len(res_lengths_res) > 0 else 0)
    res_max_length_res = torch.arange(res_max_length_val_res, device=device)
    res_key_pad_mask_res = rearrange(res_lengths_res, 'b -> b 1') <= rearrange(res_max_length_res, 'n -> 1 n')

    res_batched_feat_ids = pad_sequence(res_batched_feat_ids)
    if need_attn_mask:
        res_batched_feat_ids_1 = pad_sequence(res_batched_feat_ids_1)
        res_attn_mask = rearrange(res_batched_feat_ids, 'b i -> b 1 i 1') == rearrange(res_batched_feat_ids,
                                                                                       'b j -> b 1 1 j')
        res_attn_mask = res_attn_mask & rearrange(~res_key_pad_mask, 'b j -> b 1 1 j')
    else:
        res_attn_mask = None

    res_patches = orig_pad_sequence(res_batched_sequences, batch_first=True, padding_value=0.)
    res_patch_positions = pad_sequence(res_batched_positions) if poses is not None else None

    if pool == 'cls_token':
        res_cls_token_mask = torch.zeros_like(res_batched_feat_ids, dtype=torch.bool)
        for i in range(res_batched_feat_ids.shape[0]):
            valid_mask = ~res_key_pad_mask[i]
            feat_changes = torch.cat([
                torch.tensor([True], device=device),
                (res_batched_feat_ids[i, 1:] != res_batched_feat_ids[i, :-1]) & valid_mask[1:]
            ])
            res_cls_token_mask[i] = feat_changes & valid_mask
    else:
        res_cls_token_mask = None

    if res_key_pad_mask_no_cls is None:
        res_key_pad_mask_no_cls = res_key_pad_mask

    res_num_feats_tensor = torch.tensor(res_num_feats, device=device, dtype=torch.long)

    res_dict = {
        "patches": res_patches,
        "attn_mask": res_attn_mask,
        "key_pad_mask": res_key_pad_mask,
        "key_pad_mask_res": res_key_pad_mask_res,
        "num_feats": res_num_feats_tensor,
        "batched_feat_ids": res_batched_feat_ids,
        "cls_token_mask": res_cls_token_mask if pool == 'cls_token' else None,
        "batched_feat_ids_1": res_batched_feat_ids_1 if need_attn_mask else None,
        "key_pad_mask_no_cls": res_key_pad_mask_no_cls,
        "patch_positions": res_patch_positions,
        "batched_labels": res_batched_labels,
        "batched_num_ps": batched_num_ps
    }
    return res_dict

