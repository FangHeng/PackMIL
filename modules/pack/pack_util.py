import torch
from torch import Tensor, nn
from typing import List
import math

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


def one_hot(x, num_classes, on_value=1., off_value=0.):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)


def apply_label_activation(y: torch.Tensor, method='sharpen', activation_factor=1.0):
    """
    Map the label vector y
        - sharpen: y^(1/T)
            - temperature: T < 1 -> sharper; T > 1 -> smoother
        - balance: sigmoid(alpha * (y - mu)), where alpha = balance_c / std(y)
            - balance_c: the larger the c, the steeper; the smaller the c, the smoother
        - balance_median: sigmoid(alpha * (y - mu)), where alpha = balance_c / std(y), mu is the median of the current batch
            - balance_c: the larger the c, the steeper; the smaller the c, the smoother
        - sigmoid: sigmoid(y)
    """
    if method == 'sharpen':
        # ------ sharpen ------
        y_safe = torch.clamp(y, min=1e-8, max=1.0)
        # y^(1/T)
        y_pow = y_safe ** (1.0 / activation_factor)

        max_y_pow = torch.max(y_pow)
        if max_y_pow > 0:
            y_out = y_pow / max_y_pow
        else:
            y_out = y_pow

    elif method == 'balance':
        # ------ balance ------
        mu = torch.mean(y)
        std = torch.std(y)

        alpha = activation_factor / (std + 1e-8)

        y_shifted = y - mu
        y_scaled = alpha * y_shifted
        y_out = torch.sigmoid(y_scaled)

    elif method == 'balance_median':
        # ------ balance_median ------
        mu = torch.median(y)
        std = torch.std(y)

        alpha = activation_factor / (std + 1e-8)

        y_shifted = y - mu
        y_scaled = alpha * y_shifted
        y_out = torch.sigmoid(y_scaled)

    elif method == 'sigmoid':
        # ------ sigmoid ------
        y_out = torch.sigmoid(y)

    try:
        return y_out
    except Exception as e:
        print(f'[ERROR] {e}')
        return y


def mixup_target(target: list,
                 num_classes: int,
                 smoothing: float = 0.0,
                 multi_label: bool = True,
                 use_label_activation: bool = False,
                 activation_method: str = 'sharpen',
                 activation_factor: float = 1.0,
                 pack_num_ps: list = None,
                 target_task: str = None,
                 ):
    if target_task == 'surv':
        is_surv = True
        is_grad = False
    elif target_task == 'grade':
        is_surv = False
        is_grad = True
    else:
        is_surv = False
        is_grad = False
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value

    y_list = []

    for i, _t in enumerate(target):
        oh = one_hot(_t, num_classes, on_value=on_value, off_value=off_value)
        if pack_num_ps is not None:
            weight = pack_num_ps[i]
            oh = oh * weight

        y_list.append(oh)

    y = torch.stack(y_list, dim=0)
    y = y.squeeze(-2)

    if is_grad:
        device = y.device
        mapping_maj = torch.tensor([2.0, 3.0, 3.0, 4.0, 4.0, 5.0], dtype=torch.float32, device=device)
        mapping_min = torch.tensor([2.0, 3.0, 4.0, 3.0, 4.0, 5.0], dtype=torch.float32, device=device)

        labels_tensor = torch.tensor([int(x) for x in target], dtype=torch.int64, device=device)
        if pack_num_ps is not None:
            weights = torch.tensor(pack_num_ps, dtype=torch.float32, device=device)
        else:
            weights = torch.ones(len(labels_tensor), dtype=torch.float32, device=device)

        # calculate the weighted average for major and minor grades
        weighted_maj = (weights * mapping_maj[labels_tensor]).sum() / weights.sum()
        weighted_min = (weights * mapping_min[labels_tensor]).sum() / weights.sum()

        s = weighted_maj + weighted_min

        if weighted_min >= weighted_maj:
            grade_at_7 = 2
        else:
            grade_at_7 = 3

        #   s=0    → grade 0
        #   s=6    → grade 1
        #   s=7    → grade grade_at_7 (2 or 3)
        #   s=8    → grade 4
        #   s=9    → grade 5
        if s < 0:
            s = 0.0
        if s < 6:
            cont_grade = (s / 6) * 1.0
        elif s < 7:
            cont_grade = 1.0 + (s - 6.0) * (grade_at_7 - 1.0)
        elif s < 8:
            cont_grade = grade_at_7 + (s - 7.0) * (4.0 - grade_at_7)
        elif s < 9:
            cont_grade = 4.0 + (s - 8.0) * (5.0 - 4.0)
        else:
            cont_grade = 5.0

        lower_index = int(math.floor(cont_grade))
        upper_index = int(math.ceil(cont_grade))

        _label = torch.zeros(num_classes, dtype=torch.float32, device=device)
        if lower_index == upper_index:
            _label[lower_index] = 1.0
        else:
            _label[lower_index] = upper_index - cont_grade
            _label[upper_index] = cont_grade - lower_index

        if use_label_activation:
            _label = apply_label_activation(
                _label,
                method=activation_method,
                activation_factor=activation_factor,
            )
        return _label

    if multi_label:
        if is_surv:
            if len(target > 1):
                alpha = 0.1
                uni_target = torch.unique(target)
                alpha = min(1. / len(uni_target), alpha)
                beta = 1 - alpha * len(uni_target)

                min_idx = torch.min(uni_target)
                for i in range(num_classes):
                    if i == min_idx:
                        y[:, i] = y[:, i] * beta
                    else:
                        y[:, i] = y[:, i] * alpha

            y_sum = torch.sum(y, dim=0)  # shape = [num_classes]
            for i in range(num_classes):
                if y_sum[i] == 0:
                    for j in range(i + 1, num_classes):
                        if y_sum[j] > 0:
                            y_sum[i] = y_sum[j]
                            break
        else:
            y_sum = torch.sum(y, dim=0)  # shape = [num_classes]

        max_val = torch.max(y_sum)
        if max_val == 0:
            normalized_y = y_sum
        else:
            normalized_y = y_sum / max_val

        if use_label_activation:
            normalized_y = apply_label_activation(
                normalized_y,
                method=activation_method,
                activation_factor=activation_factor,
            )

        if is_surv:
            cumulative_y = torch.ones_like(normalized_y)
            for i in range(num_classes):
                if i == 0:
                    cumulative_y[i] = normalized_y[i]
                else:
                    cumulative_y[i] = cumulative_y[i - 1] * normalized_y[i]
            return cumulative_y / torch.sum(cumulative_y)
        return normalized_y
    else:
        y_sum = torch.sum(y, dim=0)  # shape = [num_classes]
        # Normalize probabilities to be between 0-1 and sum to 1
        if torch.sum(y_sum) > 0:
            normalized_y = y_sum / torch.sum(y_sum)
        else:
            # Handle the edge case where all values are zero
            normalized_y = torch.ones_like(y_sum) / y_sum.size(0)
        return normalized_y


def _argmin(tensor):
    _min_value = tensor[0]
    _min_index = 0

    for i in range(len(tensor)):
        if tensor[i] < _min_value:
            _min_value = tensor[i]
            _min_index = i
    return _min_index


def mixup_target_batched(target, num_classes, batched_num_ps=None, target_task=None, **kwargs):
    if target_task == 'surv':
        kwargs_c = kwargs.copy()
        kwargs_c['multi_label'] = False
        kwargs['is_surv'] = True
        if kwargs['multi_label']:
            y_batched = torch.empty(len(target), num_classes, device=target[0].device)
            y_censored_batched = torch.empty(len(target), num_classes, device=target[0].device)
            c_batched = torch.empty(len(target), 1, device=target[0].device)
            for i, _t in enumerate(target):
                _y = [target[i][_i][0].unsqueeze(0) for _i in range(len(target[i]))]
                _y = torch.cat(_y)
                _c = [target[i][_i][1] for _i in range(len(target[i]))]

                non_censored_indices = [j for j, cens in enumerate(_c) if cens == 0]
                censored_indices = [j for j, cens in enumerate(_c) if cens == 1]

                non_censored_indices = non_censored_indices if len(non_censored_indices) > 0 else censored_indices
                censored_indices = censored_indices if len(censored_indices) > 0 else non_censored_indices

                _y_uncensored = _y[non_censored_indices]
                _y_censored = _y[censored_indices]

                y_batched[i] = mixup_target(target=_y_uncensored, num_classes=num_classes,
                                            pack_num_ps=batched_num_ps[i][non_censored_indices],
                                            **kwargs) if batched_num_ps is not None else mixup_target(_y_uncensored,
                                                                                                      num_classes,
                                                                                                      **kwargs)

                y_censored_batched[i] = mixup_target(target=_y_censored, num_classes=num_classes,
                                                     pack_num_ps=batched_num_ps[i][censored_indices],
                                                     **kwargs) if batched_num_ps is not None else mixup_target(
                    _y_censored, num_classes, **kwargs)

                _tmp = mixup_target(target=_c, num_classes=2, pack_num_ps=batched_num_ps[i],
                                    **kwargs_c) if batched_num_ps is not None else mixup_target(_c, 2, **kwargs_c)
                c_batched[i] = _tmp[1]
        else:
            y_batched = torch.empty(len(target), 1, device=target[0].device)
            y_censored_batched = torch.empty(len(target), 1, device=target[0].device)
            c_batched = torch.empty(len(target), 1, device=target[0].device)
            for i, _t in enumerate(target):
                _y = [target[i][_i][0] for _i in range(len(target[i]))]
                _c = [target[i][_i][1] for _i in range(len(target[i]))]
                idx_min = _argmin(_y)
                if _c[idx_min] == 0:
                    y_batched[i] = _y[idx_min]
                    c_batched[i] = _c[idx_min]
                    y_censored_batched[i] = y_batched[i]

                else:
                    non_censored_indices = [j for j, cens in enumerate(_c) if cens == 0]
                    y_censored_batched[i] = _y[idx_min]
                    if non_censored_indices:
                        min_non_censored_idx = non_censored_indices[_argmin([_y[j] for j in non_censored_indices])]
                        y_batched[i] = _y[min_non_censored_idx]
                    else:
                        y_batched[i] = _y[idx_min]

                    _tmp = mixup_target(target=_c, num_classes=2,
                                        pack_num_ps=batched_num_ps[i],
                                        **kwargs_c) if batched_num_ps is not None else mixup_target(_c, 2, **kwargs_c)
                    c_batched[i] = _tmp[1]
        y_batched = {
            'Y': y_batched,
            'c': c_batched,
            'Y_censored': y_censored_batched,
        }

    else:
        y_batched = torch.empty(len(target), num_classes, device=target[0].device)
        for i, _t in enumerate(target):
            y_batched[i] = mixup_target(target=target[i],
                                        num_classes=num_classes,
                                        pack_num_ps=batched_num_ps[i] if batched_num_ps is not None else None,
                                        target_task=target_task,
                                        **kwargs)

    return y_batched


def get_seq_pos(coords, patch_size, patch_level):
    patch_x = coords[0] // patch_size[0]  # Horizontal coordinate
    patch_y = coords[1] // patch_size[1]  # Vertical coordinate

    all_x = patch_level[0] // patch_size[0]
    all_y = patch_level[1] // patch_size[1]

    return all_x, all_y


def get_dropout(
        seqs: List[Tensor],
        base_token_dp: float = 0.,
        _type: str = 'random',
):
    n_seqs = len(seqs)
    device = seqs[0].device

    if _type == 'random':
        std = 0.015
        means = torch.full((n_seqs,), base_token_dp, device=device)
        samples = torch.normal(means, std)

        lower_bound = means - 2 * std
        upper_bound = means + 2 * std
        samples = torch.clamp(samples, lower_bound, upper_bound)

        return samples

    else:
        raise ValueError(f"Unknown dropout type: {_type}")

def group_seqs(seqs: List[torch.Tensor],
               poses=None,
               labels=None,
               token_dropout=0.,
               max_seqs_len=2048,
               min_seq_len=512,
               grouping_strategy="sequential",
               return_indices=False,
               additional_token_drop=None
               ):
    """
    Pack the sequence into groups according to the length limit.
    If return_indices is True, the corresponding original sequence index in each group is also returned for subsequent indexing operations.
    """
    groups = []
    groups_ps = []
    groups_indices = [] if return_indices else None
    token_dropout = [token_dropout for _ in range(len(seqs))] if not isinstance(token_dropout,
                                                                                torch.Tensor) else token_dropout
    seq_lengths = [seq.shape[0] for seq in seqs]
    if grouping_strategy == "sequential":
        group = []
        group_ps = []
        group_idx = [] if return_indices else None
        seqs_len = 0
        for i, seq in enumerate(seqs):
            seq_len = seq_lengths[i]
            if seq_len > min_seq_len:
                if additional_token_drop is not None:
                    seq_len = int(int(seq_len * (1 - token_dropout[i])) / additional_token_drop)
                else:
                    seq_len = int(seq_len * (1 - token_dropout[i]))
                seq_len = max(seq_len, min_seq_len)
                seq_len = min(seq_len, max_seqs_len)
            assert seq_len <= max_seqs_len, f'sequence with dimensions {seq_len} exceeds maximum sequence length'

            if (seqs_len + seq_len) > max_seqs_len:
                groups.append(group)
                groups_ps.append(group_ps)
                if return_indices:
                    groups_indices.append(group_idx)
                group = []
                group_ps = []
                group_idx = [] if return_indices else None
                seqs_len = 0

            seq_data = [seq, labels[i]] if labels is not None else seq
            if poses is not None:
                group.append([seq_data, token_dropout[i], poses[i]])
            else:
                group.append([seq_data, token_dropout[i]])
            if return_indices:
                group_idx.append(i)
            group_ps.append(seq_len)
            seqs_len += seq_len

        if len(group) > 0:
            groups.append(group)
            groups_ps.append(group_ps)
            if return_indices:
                groups_indices.append(group_idx)

    else:
        raise ValueError(f"Unknown grouping strategy: {grouping_strategy}")

    if return_indices:
        return groups, groups_ps, groups_indices
    else:
        return groups, groups_ps