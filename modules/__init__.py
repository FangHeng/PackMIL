import torch


from .abmil import DAttention,AttentionGated
from .clam import CLAM_MB,CLAM_SB
from .dsmil import MILNet
from .transmil import TransMIL
from .mean_max import MeanMIL,MaxMIL
from .dtfd import DTFD
from .rrt import RRTMIL
from .wikg import WiKG
from .packmil import PackMIL
from .vit_mil import ViTMIL
from modules.pack.pack_baseline import MILBase
from .mambamil_2d import MambaMIL_2D
from .gigap import GIGAPMIL
from .chief import CHIEF
from .utils import get_mil_model_params,get_mil_model_params_from_name

import os 

def build_model(args,device):
    if args.pack_bs:
        others = {}
        _,genera_trans_params = get_mil_model_params(args)
        genera_trans_params.update(get_mil_model_params_from_name(args,args.model))
        task_type = 'surv' if args.datasets.lower().startswith('surv') else 'grade' if 'panda' in args.datasets.lower() else 'subtype'
        model = PackMIL(
            mil=args.model,
            task_type=task_type,
            token_dropout=args.token_dropout,
            group_max_seq_len=args.pack_max_seq_len,
            min_seq_len=args.min_seq_len,
            pack_residual=not args.pack_no_residual,
            residual_loss=args.pack_residual_loss,
            downsample_mode=args.pack_downsample_mode,
            downsample_type=args.pack_downsample_type,
            residual_type=args.pack_residual_type,
            residual_ps_weight=args.pack_residual_ps_weight,
            singlelabel=args.pack_singlelabel,
            downsample_r=args.pack_residual_downsample_r,
            pad_r=args.pack_pad_r,
            epeg_k=args.epeg_k,
            crmsa_k=args.crmsa_k,
            region_num=args.region_num,
            **genera_trans_params
            ).to(device)

        return model,others
    else:
        return build_mil(args,args.model,device)

def build_mil(args,model_name,device):
    others = {}

    if args.teacher_init is not None:
        if not args.teacher_init.endswith('.pt'):
            _str = 'fold_{fold}_model_best.pt'.format(fold=args.fold_curr)
            _teacher_init = os.path.join(args.teacher_init,_str)
        else:
            _teacher_init = args.teacher_init

    genera_model_params,genera_trans_params = get_mil_model_params(args)

    if model_name == 'rrtmil':
        model = RRTMIL(input_dim=args.input_dim,n_classes=args.n_classes,epeg_k=args.epeg_k,crmsa_k=args.crmsa_k,region_num=args.region_num,n_heads=args.rrt_n_heads,n_layers=args.rrt_n_layers,mil_norm=args.mil_norm,embed_norm_pos=args.embed_norm_pos).to(device)
    elif model_name == 'wikg':
        model = WiKG(dim_in=args.input_dim, topk=args.wikg_topk, n_classes=args.n_classes, agg_type=args.wikg_agg_type, dropout=args.dropout, pool=args.wikg_pool).to(device)
    elif model_name == 'abmil':
        model = DAttention(**genera_model_params).to(device)
    elif model_name == 'gabmil':
        model = AttentionGated(**genera_model_params).to(device)
    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    elif model_name == 'clam_sb':
        model = CLAM_SB(**genera_model_params).to(device)
    elif model_name == 'clam_mb':
        model = CLAM_MB(**genera_model_params).to(device)
    elif model_name == 'transmil':
        model = TransMIL(**genera_trans_params).to(device)
    elif model_name == 'vitmil':
        model = ViTMIL(**genera_trans_params).to(device)
    elif model_name == 'dsmil':
        model = MILNet(**genera_model_params).to(device)
        if args.aux_alpha == 0.:
            args.main_alpha = 0.5
            args.aux_alpha = 0.5
        #state_dict_weights = torch.load('./modules/init_cpk/dsmil_init.pth')
        #info = model.load_state_dict(state_dict_weights, strict=False)
    elif model_name == 'dtfd':
        model = DTFD(device=device, lr=args.lr, weight_decay=args.weight_decay, steps=args.num_epoch, input_dim=args.input_dim, n_classes=args.n_classes).to(device)
    elif model_name == 'meanmil':
        model = MeanMIL(**genera_model_params).to(device)
    elif model_name == 'maxmil':
        model = MaxMIL(**genera_model_params).to(device)
    elif model_name == '2dmamba':
        model_params = {
            'in_dim': args.input_dim,
            'n_classes': args.n_classes,
            'mambamil_dim': args.mambamil_dim,
            'mambamil_layer': args.mambamil_layer,
            'mambamil_state_dim': args.mambamil_state_dim,
            'pscan': args.pscan,
            'cuda_pscan': args.cuda_pscan,
            'mamba_2d_max_w': args.mamba_2d_max_w,
            'mamba_2d_max_h': args.mamba_2d_max_h,
            'mamba_2d_pad_token': args.mamba_2d_pad_token,
            'mamba_2d_patch_size': args.mamba_2d_patch_size,
            'pos_emb_type': args.mamba_2d_pos_emb_type,
            'drop_out': args.dropout,
            'pos_emb_dropout': args.pos_emb_dropout,
        }
        model = MambaMIL_2D(**model_params).to(device)
    elif model_name == 'pack_pure':
        if args.baseline == 'abmil':
            from modules.pack.pack_baseline import DAttention as DAAggregate
            model = MILBase(aggregate_fn=DAAggregate, **genera_model_params).to(device)
        elif args.baseline == 'dsmil':
            from modules.pack.pack_baseline import DSAttention as DSAggregate
            model = MILBase(aggregate_fn=DSAggregate,agg_n_classes=args.n_classes, **genera_model_params).to(device)
        elif args.baseline == 'vitmil':
            from modules.pack.pack_baseline import SAttention as SAAggregate
            model = MILBase(aggregate_fn=SAAggregate, **genera_model_params).to(device)
    elif model_name == 'gigap':
        model = GIGAPMIL(**genera_model_params).to(device)
    elif model_name == 'chief':
        model = CHIEF(**genera_model_params,dataset=args.datasets.lower()).to(device)
        if 'CHIEF_MIL_PATH' not in os.environ or not os.environ['CHIEF_MIL_PATH']:
            os.environ['CHIEF_MIL_PATH'] = 'xxx/dataset/wsi_data/ckp/chief/CHIEF_pretraining.pth'
        chief_mil_path = os.environ.get('CHIEF_MIL_PATH')
        if chief_mil_path and os.path.exists(chief_mil_path):
            state_dict = torch.load(chief_mil_path, map_location="cpu")
            for key in ['classifiers.weight', 'classifiers.bias']:
                if key in state_dict:
                    del state_dict[key]
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                for k in missing_keys:
                    print("Missing ", k)
            if unexpected_keys:
                for k in unexpected_keys:
                    print("Unexpected ", k)
        else:
            print("[INFO] CHIEF_WEIGHT is not set. Skipping initialization.")
    else:
        raise NotImplementedError
    
    return model, others