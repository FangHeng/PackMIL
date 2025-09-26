import argparse
import yaml
import torch
from timm.utils import init_distributed_device

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Computional Pathology Training Script')

##### Dataset 
group = parser.add_argument_group('Dataset')
# Paths
parser.add_argument('--dataset_root', default='/data/xxx/TCGA', type=str, help='Dataset root path')
group.add_argument('--csv_path', default=None, type=str, help='Dataset CSV path Label and Split')
group.add_argument('--h5_path', default=None, type=str, help='Dataset H5 path. Coord.')
# Dataset settings
group.add_argument('--datasets', default='brca', type=str, help='[brca, panda, nsclc, luad, call, surv_brca, surv_luad]')
group.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio')
group.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
group.add_argument('--cv_fold', default=5, type=int, help='Number of cross validation fold [5]')
group.add_argument('--random_seed_fold', default=0, type=int, help='Random seed used when creating cross-validation folds')
group.add_argument('--val2test', action='store_true', help='Use validation set as test set')
group.add_argument('--random_fold', action='store_true', help='Muti-fold random experiment')
# Patch size settings
group.add_argument('--same_psize', default=0., type=float, help='Keep the same size of all patches [0]')
group.add_argument('--same_psize_pad_type', default='zero', type=str, choices=['zero', 'random', 'none'])
group.add_argument('--same_psize_ratio', default=0., type=float, help='Keep the same ratio of all patches [0]')
# Dataloader settings
group.add_argument('--num_workers', default=6, type=int, help='Number of workers in the dataloader')
group.add_argument('--num_workers_test', default=None, type=int, help='Number of workers in the dataloader')
group.add_argument('--pin_memory', action='store_true', help='Enable Pinned Memory')
group.add_argument('--file_worker', action='store_true', help='Enable file system sharing via workers')
group.add_argument('--no_prefetch', action='store_true', help='Disable prefetching')
group.add_argument('--prefetch_factor', default=2, type=int, help='Prefetch factor [2]')
group.add_argument('--no_val_ddp', action='store_true', help='Disable DDP for validation')
group.add_argument('--persistence', action='store_true', help='Preload feature tensors so they persist in memory across iterations')

##### Train
group = parser.add_argument_group('Training')
group.add_argument('--main_alpha', default=1.0, type=float, help='Main loss alpha')
group.add_argument('--num_epoch', default=200, type=int, help='Number of total training epochs [200]')
group.add_argument('--epoch_start', default=0, type=int, help='Epoch index to resume training from')
group.add_argument('--early_stopping', action='store_false', help='Early stopping')
group.add_argument('--max_epoch', default=130, type=int, help='Number of max training epochs in the earlystopping [130]')
group.add_argument('--warmup_epochs', default=0, type=int, help='Number of training epochs with warmup lr')
group.add_argument('--patient', default=20, type=int, help='Patience (in epochs) before early stopping is triggered')
group.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
group.add_argument('--loss', default='ce', type=str, choices=['ce', 'bce', 'asl', 'nll_surv'], help='Classification Loss, defualt nll_surv in survival prediction [ce, bce, nll_surv]')
group.add_argument('--label_smooth', default=0., type=float, help='Label smoothing factor')
group.add_argument('--opt', default='adam', type=str, help='Optimizer [adam, adamw]')
group.add_argument('--model', default='abmil', type=str, help='Model name')
group.add_argument('--seed', default=2021, type=int, help='random number [2021]')
group.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
group.add_argument('--lr_base_size', default=1, type=int, help='Reference batch size used for learning-rate scaling')
group.add_argument('--warmup_lr', default=1e-6, type=float, help='Starting learning rate during the warmup phase')
group.add_argument('--lr_sche', default='cosine', type=str, help='Decay of learning rate [cosine, step, const]')
group.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
group.add_argument('--lr_scale', action='store_true', help='Scale learning rate according to global-to-base batch size ratio')
group.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
group.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulation steps')
group.add_argument('--always_test', action='store_true', help='Test model in the training phase')
group.add_argument('--model_ema', action='store_true', help='Enable Model EMA')
group.add_argument('--bin_metric', action='store_true', help='Use binary average when n_classes==2')
group.add_argument('--no_determ', action='store_true', help='Disable PyTorch deterministic mode (enables CuDNN benchmark)')
group.add_argument('--no_deter_algo', action='store_true', help='Allow non-deterministic algorithms when determinism is enabled')
group.add_argument('--no_drop_last', action='store_true', default=False, help='Keep the last incomplete batch instead of dropping it')
group.add_argument('--empty_cuda_cache', action='store_true', default=False, help='Clear CUDA cache')
group.add_argument('--aux_alpha', default=0., type=float, help='Auxiliary loss alpha')
group.add_argument('--test_type', default='main', type=str, choices=['main', 'ema', 'both', 'both_ema'])

##### Evaluate
group = parser.add_argument_group('Evaluate')
group.add_argument('--all_test', action='store_true', help='Evaluate using the union of train, validation, and test splits')
group.add_argument('--num_bootstrap', default=1000, type=int, help='Number of bootstrap samples for metric estimation')
group.add_argument('--bootstrap_mode', default='test', type=str, choices=['test', 'none', 'val', 'test_val'])

##### Model
group = parser.add_argument_group('Model')
# General
group.add_argument('--input_dim', default=1024, type=int, help='dim of input features. PLIP features should be [512]')
group.add_argument('--n_classes', default=2, type=int, help='Number of classes')
group.add_argument('--act', default='relu', type=str, choices=['relu', 'gelu', 'none'],
                   help='Activation func in the projection head [gelu,relu]')
group.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
group.add_argument('--mil_norm', default=None, choices=['bn', 'ln', 'none'])
group.add_argument('--cls_norm', default=None, choices=['bn', 'ln', 'none'])
group.add_argument('--no_mil_bias', action='store_true', help='DSMIL hyperparameter')
group.add_argument('--inner_dim', default=512, type=int, help='Hidden dimension for the MIL backbone')
group.add_argument('--mil_feat_embed_mlp_ratio', default=4, type=int, help='Expansion ratio for the MIL feature embedding MLP')
group.add_argument('--embed_norm_pos', default=0, type=int, help='Position of normalization in the feature embed (0=input, 1=hidden)')
# Shuffle
group.add_argument('--patch_shuffle', action='store_true', help='Patch shuffle')
# Common MIL models
group.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')
group.add_argument('--da_gated', action='store_true', help='DSMIL hyperparameter')
# General Trans
group.add_argument('--pos', default=None, type=str, choices=['ppeg', 'sincos', 'alibi', 'none', 'alibi_learn', 'rope'])
group.add_argument('--n_heads', default=8, type=int, help='Number of head in the MSA')
group.add_argument('--n_layers', default=2, type=int, help='Number of transformer layers in the MIL backbone')
group.add_argument('--pool', default='cls_token', type=str, help='Pooling strategy applied to MIL encoder outputs')
group.add_argument('--attn_dropout', default=0., type=float, help='Dropout rate applied inside attention blocks')
group.add_argument('--ffn', action='store_true', help='DSMIL hyperparameter')
group.add_argument('--sdpa_type', default='torch', type=str, choices=['torch', 'flash', 'math', 'memo_effi', 'torch_math', 'ntrans'])
group.add_argument('--attn_type', default='sa', type=str, choices=['sa', 'ca', 'ntrans'], help='Type of attention mechanism to use in transformer blocks. [Only for Ablation]')

##### WiKG
group = parser.add_argument_group('WiKG')
group.add_argument('--wikg_topk', default=6, type=int, help='Topk of the WiKG')
group.add_argument('--wikg_agg_type', default='bi-interaction', type=str, choices=['bi-interaction', 'sage', 'gcn'])
group.add_argument('--wikg_pool', default='attn', type=str, help='Pool of the WiKG', choices=['attn', 'mean', 'max'])

##### PackMIL
group = parser.add_argument_group('PackMIL')
group.add_argument('--pack_bs', action='store_true', help='use packmil')
group.add_argument('--token_dropout', default=0., type=float, help='Drop ratio applied when sampling tokens for packing')
group.add_argument('--pack_max_seq_len', default=5120, type=int, help='Maximum packed sequence length per bag')
group.add_argument('--min_seq_len', default=128, type=int, help='Minimum sequence length enforced after packing')
group.add_argument('--no_norm_pad', action='store_true', help='Skip normalization on padded tokens when packing')
group.add_argument('--pack_residual', action='store_true', help='Enable residual branch to learn from dropped tokens')
group.add_argument('--pack_residual_loss', default="bce", type=str, choices=['bce', 'asl', 'ce', 'nll', 'asl_single', 'focal'])
group.add_argument('--pack_residual_ps_weight', action='store_true', help='Weight residual targets by the number of kept patches')
group.add_argument('--pack_residual_downsample_r', default=None, type=int, help='Downsample ratio used by the residual branch')
group.add_argument('--pack_downsample_mode', default="ads", type=str, choices=['no_pu','ads'], help='Strategy used to downsample sequences inside PackMIL')
group.add_argument('--pack_singlelabel', action='store_true', help='Treat residual supervision as single-label targets')

##### RRT
group = parser.add_argument_group('RRT')
group.add_argument('--epeg_k', default=15, type=int, help='Number of the epeg_k')
group.add_argument('--crmsa_k', default=3, type=int, help='Number of the crmsa_k')
group.add_argument('--region_num', default=8, type=int, help='region num')
group.add_argument('--rrt_n_heads', default=8, type=int, help='Number of heads')
group.add_argument('--rrt_n_layers', default=2, type=int, help='Number of transformer layers in RRT blocks')
group.add_argument('--rrt_pool', default="attn", type=str)

##### Mamba
group = parser.add_argument_group('Mamba')
parser.add_argument('--mambamil_dim', type=int, default=128)
parser.add_argument('--mambamil_rate', type=int, default=10)
parser.add_argument('--mambamil_state_dim', type=int, default=16)
parser.add_argument('--mambamil_layer', type=int, default=1)
parser.add_argument('--mambamil_inner_layernorms', default=False, action='store_true')
parser.add_argument('--mambamil_type', type=str, default=None, choices=['Mamba', 'SRMamba', 'SimpleMamba'],
                    help='Variant of the Mamba backbone to instantiate')
parser.add_argument('--pscan', default=True, help='Enable prefix-scan optimizations when available')
parser.add_argument('--cuda_pscan', default=False, action='store_true')
parser.add_argument('--pos_emb_dropout', type=float, default=0.0)
parser.add_argument('--mamba_2d', default=False, action='store_true')
parser.add_argument('--mamba_2d_pad_token', '-p', type=str, default='trainable', choices=['zero', 'trainable'])
parser.add_argument('--mamba_2d_patch_size', type=int, default=1)
parser.add_argument('--mamba_2d_pos_emb_type', default=None, choices= [None, 'linear'])

##### Misc
group = parser.add_argument_group('Miscellaneous')
group.add_argument('--title', default='default', type=str, help='Title of exp')
group.add_argument('--project', default='mil_new_c16', type=str, help='Project name of exp')
group.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
group.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
group.add_argument('--amp_test', action='store_true', help='Automatic Mixed Precision Training')
group.add_argument('--amp_scale_index', default=16, type=int, help='Automatic Mixed Precision Training')
group.add_argument('--amp_growth_interval', default=2000, type=int, help='Automatic Mixed Precision Training')
group.add_argument('--amp_unscale', action='store_true', help='Automatic Mixed Precision Training')
group.add_argument('--output_path', type=str, help='Output path')
group.add_argument('--model_path', default=None, type=str, help='model init path')
group.add_argument("--local-rank", "--local_rank", type=int)
group.add_argument('--script_mode', default='all', type=str, help='[all, no_train, test, only_train]')
group.add_argument('--profile', action='store_true', help='Enable torch.profiler tracing during training iterations')
# Torchcompile
group.add_argument('--torchcompile', action='store_true', help='Torch Compile for torch > 2.0')
group.add_argument('--torchcompile_mode', default='default', type=str,
                   choices=['default', 'reduce-overhead', 'max-autotune'], help='Compilation mode to pass into torch.compile')
# Wandb
group.add_argument('--wandb', action='store_true', help='Log metrics to Weights & Biases')
group.add_argument('--wandb_watch', action='store_true', help='Watch model parameters/gradients in Weights & Biases')
# DDP
group.add_argument('--no_ddp_broad_buf', action='store_true', help='Disable broadcast_buffers in DistributedDataParallel')
group.add_argument('--save_iter', default=-1, type=int, help='Checkpoint saving interval in epochs (-1 to disable)')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()

    cfg = {}
    if args_config.config:
        config_files = args_config.config.split(',')

        for config_file in config_files:
            config_file = config_file.strip()  
            if config_file: 
                try:
                    with open(config_file, 'r') as f:
                        cfg.update(yaml.safe_load(f))
                except Exception as e:
                    print(f"Error loading config file {config_file}: {str(e)}")

        parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    args.config = args_config.config.split(',')

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    return args, args_text


def more_about_config(args):
    # more about config

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device = init_distributed_device(args)
    # train
    args.mil_bias = not args.no_mil_bias
    args.drop_last = not args.no_drop_last
    args.prefetch = not args.no_prefetch
    args.mil_feat_embed = True
    args.mil_feat_embed_type = "norm"
    args.seed_ori = args.seed
    if not args.amp_test:
        args.amp_test = args.amp

    if args.pos in ('sincos', 'alibi', 'alibi_learn', 'rope'):
        assert args.h5_path is not None

    if args.persistence:
        if args.same_psize > 0:
            raise NotImplementedError("Random same patch is different from not presistence")

    if args.val2test or args.val_ratio == 0.:
        args.always_test = False

    if args.pos in ('sincos', 'alibi') and args.h5_path is None:
        raise NotImplementedError

    if args.mil_feat_embed:
        args.mil_feat_embed = ~(args.mil_feat_embed_type == 'none')

    if args.datasets.lower() == 'panda':
        args.n_classes = 6

    if args.model == 'mhim_pure':
        args.aux_alpha = 0.
    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    elif args.model in ('clam_sb', "clam_mb"):
        args.main_alpha = .7
        args.aux_alpha = .3
    elif args.model == 'dsmil':
        if args.main_alpha > 0.:
            args.main_alpha = 0.5
            args.aux_alpha_1 = 0.5
        else:
            args.aux_alpha_1 = 0.
        args.aux_alpha_2 = args.aux_alpha
        args.aux_alpha = 1

    if args.model == '2dmamba':
        if args.datasets.lower().endswith('brca'):
            args.mamba_2d_max_w = 413
            args.mamba_2d_max_h = 821
        elif args.datasets.lower().endswith('panda'):
            args.mamba_2d_max_w = 384
            args.mamba_2d_max_h = 216
        elif args.datasets.lower().endswith('nsclc') or args.datasets.lower().endswith('luad') or args.datasets.lower().endswith('lusc'):
            args.mamba_2d_max_w = 385
            args.mamba_2d_max_h = 216
        elif args.datasets.lower().endswith('call'):
            args.mamba_2d_max_w = 432
            args.mamba_2d_max_h = 432
        elif args.datasets.lower().endswith('blca'):
            args.mamba_2d_max_w = 381
            args.mamba_2d_max_h = 275
        else:
            raise NotImplementedError(args.datasets)

    # multi-class cls, refer to top-1 acc, bin-class cls, refer to auc
    args.best_metric_index = 1 if args.n_classes != 2 and not args.datasets.lower().startswith('surv') else 0

    args.max_epoch = min(args.max_epoch, args.num_epoch)

    return args, device