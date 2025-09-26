import time
import torch
import wandb
import os
import numpy as np
import random
import traceback
from einops._torch_specific import allow_ops_in_compiled_graph
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from timm.utils import AverageMeter,distribute_bn
from timm.layers import convert_sync_batchnorm
from collections import OrderedDict
import gc

from datasets import build_dataloader,get_kfold,get_patient_label,get_patient_label_surv
from datasets.data_utils import get_split_dfs,get_data_dfs
import utils
from modules import build_model
from train_utils import build_train
from engines import build_engine
from options import _parse_args,more_about_config

def main(args,device):
    # set seed
    utils.seed_torch(args.seed)

    label_path = args.csv_path if args.csv_path else os.path.join(args.dataset_root,'label.csv')
    if args.cv_fold > 1 and not os.path.isdir(label_path) and not args.random_fold:
        if args.datasets.lower().startswith('surv'):
            df = get_patient_label_surv(args,label_path)
        else:
            df = get_patient_label(args,label_path)

        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

        if args.cv_fold > 1:
            train_dfs, test_dfs, val_dfs = get_kfold(args,args.cv_fold, df, val_ratio=args.val_ratio)

    if args.datasets.lower().startswith('surv'):
        cindex,cindex_std, te_cindex = [],[],[]
        cindex_ema = cindex_ema_std = []
        ckc_metric = [cindex,cindex_std]
        ckc_metric_ema = [cindex_ema,cindex_ema_std]
        te_ckc_metric = [te_cindex]
    else:
        acc, pre, rec,fs,auc,ck,acc_m,te_auc,te_fs=[],[],[],[],[],[],[],[],[]
        acc_std,auc_std,fs_std,ck_std,acc_m_std = [],[],[],[],[]
        acc_ema,pre_ema,rec_ema,fs_ema,auc_ema,acc_ema_std,auc_ema_std,fs_ema_std,ck_ema,acc_m_ema,ck_ema_std,acc_m_ema_std = [],[],[],[],[],[],[],[],[],[],[],[]
        ckc_metric = [auc,acc, pre, rec,fs,ck,acc_m,auc_std,acc_std,fs_std,ck_std,acc_m_std]
        ckc_metric_ema = [auc_ema,acc_ema, pre_ema, rec_ema,fs_ema,ck_ema,acc_m_ema,auc_ema_std,acc_ema_std,fs_ema_std,ck_ema_std,acc_m_ema_std]
        te_ckc_metric = [te_auc,te_fs]

    if not args.no_log and args.rank == 0:
        print('Dataset: ' + args.datasets)

    for k in range(args.fold_start, args.cv_fold):
        if args.cv_fold > 1 and not os.path.isdir(label_path) and not args.random_fold:
            dataset = {
                'train': train_dfs[k],
                'test': test_dfs[k],
                'val': val_dfs[k]
            }
        else:
            _label_path = os.path.join(args.csv_path,f'fold_{k}.csv') if os.path.isdir(label_path) else args.csv_path
            df = get_data_dfs(args, _label_path)
            train_dfs, test_dfs, val_dfs = get_split_dfs(args,df)

            dataset = {
                'train': train_dfs,
                'test': test_dfs,
                'val': val_dfs
            }

        args.fold_curr = k
        if not args.no_log and args.rank == 0:
            print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
        ckc_metric,te_ckc_metric,ckc_metric_ema = one_fold(args,device,ckc_metric,te_ckc_metric,ckc_metric_ema,dataset)

    # delete cpk
    if args.rank == 0:
        if os.path.isfile(os.path.join(args.output_path,'ckp.pt')):
            os.remove(os.path.join(args.output_path,'ckp.pt'))

    if args.random_seed_fold > 0:
        if args.always_test:
            raise NotImplementedError('random_seed_fold and always_test are not supported in this time')
        if args.datasets.lower().startswith('surv'):
            return cindex,cindex_std,cindex_ema
        else:
            return (acc, auc, fs, pre, rec, ck, acc_m, auc_std, acc_std,  fs_std,
                    acc_ema, auc_ema, fs_ema, pre_ema, rec_ema, ck_ema, acc_m_ema, auc_ema_std,
                    acc_ema_std, fs_ema_std, acc_m_ema_std)

    if args.always_test:
        if args.wandb and args.rank == 0:
            if args.datasets.lower().startswith('surv'):
                wandb.log({
                    "cross_val/te_cindex_mean":np.mean(np.array(te_cindex)),
                    "cross_val/te_cindex_std":np.std(np.array(te_cindex)),
                })
            else:
                wandb.log({
                    "cross_val/te_auc_mean":np.mean(np.array(te_auc)),
                    "cross_val/te_auc_std":np.std(np.array(te_auc)),
                    "cross_val/te_f1_mean":np.mean(np.array(te_fs)),
                    "cross_val/te_f1_std":np.std(np.array(te_fs)),
                })
    

    if args.wandb and args.rank == 0:
        if args.datasets.lower().startswith('surv'):
            wandb.log({
                "cross_val/cindex_mean":np.mean(np.array(cindex)),
                "cross_val/cindex_std_mean":np.mean(np.array(cindex_std)),
                "cross_val/cindex_std":np.std(np.array(cindex)),
            })
        else:
            wandb.log({
                "cross_val/acc_mean":np.mean(np.array(acc)),
                "cross_val/auc_mean":np.mean(np.array(auc)),
                "cross_val/f1_mean":np.mean(np.array(fs)),
                "cross_val/acc_std_mean":np.mean(np.array(acc_std)),
                "cross_val/auc_std_mean":np.mean(np.array(auc_std)),
                "cross_val/f1_std_mean":np.mean(np.array(fs_std)),
                "cross_val/pre_mean":np.mean(np.array(pre)),
                "cross_val/recall_mean":np.mean(np.array(rec)),
                "cross_val/ck_mean":np.mean(np.array(ck)),
                "cross_val/acc_micro_mean":np.mean(np.array(acc_m)),
                "cross_val/acc_std":np.std(np.array(acc)),
                "cross_val/auc_std":np.std(np.array(auc)),
                "cross_val/f1_std":np.std(np.array(fs)),
                "cross_val/pre_std":np.std(np.array(pre)),
                "cross_val/recall_std":np.std(np.array(rec)),
                "cross_val/ck_std":np.std(np.array(ck)),
                "cross_val/acc_micro_std":np.std(np.array(acc_m)),
            })
    
    if args.wandb and args.rank == 0:
        if args.datasets.lower().startswith('surv'):
            wandb.log({
                "cross_val/cindex_ema_mean":np.mean(np.array(cindex_ema)),
                "cross_val/cindex_ema_std_mean":np.mean(np.array(cindex_ema_std)),
                "cross_val/cindex_ema_std":np.std(np.array(cindex_ema)),
            })
        else:
            wandb.log({
                "cross_val/acc_ema_mean":np.mean(np.array(acc_ema)),
                "cross_val/auc_ema_mean":np.mean(np.array(auc_ema)),
                "cross_val/f1_ema_mean":np.mean(np.array(fs_ema)),
                "cross_val/ck_ema_mean":np.mean(np.array(ck_ema)),
                "cross_val/acc_micro_ema_mean":np.mean(np.array(acc_m_ema)),
                "cross_val/acc_ema_std_mean":np.mean(np.array(acc_ema_std)),
                "cross_val/auc_ema_std_mean":np.mean(np.array(auc_ema_std)),
                "cross_val/f1_ema_std_mean":np.mean(np.array(fs_ema_std)),
                "cross_val/ck_ema_std_mean":np.mean(np.array(ck_ema_std)),
                "cross_val/acc_micro_ema_std_mean":np.mean(np.array(acc_m_ema_std)),
                "cross_val/pre_ema_mean":np.mean(np.array(pre_ema)),
                "cross_val/recall_ema_mean":np.mean(np.array(rec_ema)),
                "cross_val/acc_ema_std":np.std(np.array(acc_ema)),
                "cross_val/auc_ema_std":np.std(np.array(auc_ema)),
                "cross_val/f1_ema_std":np.std(np.array(fs_ema)),
                "cross_val/pre_ema_std":np.std(np.array(pre_ema)),
                "cross_val/recall_ema_std":np.std(np.array(rec_ema)),
                "cross_val/ck_ema_std":np.std(np.array(ck_ema)),
                "cross_val/acc_micro_ema_std":np.std(np.array(acc_m_ema)),
            })

    if not args.no_log and args.rank == 0:
        if args.datasets.lower().startswith('surv'):
            print('Cross validation c-index mean: %.4f, std %.4f ' % (np.mean(np.array(cindex)), np.std(np.array(cindex))))
            if args.test_type != 'main' or args.model_ema:
                pass
        else:
            print('Cross validation accuracy mean: %.4f, std %.4f ' % (np.mean(np.array(acc)), np.std(np.array(acc))))
            print('Cross validation auc mean: %.4f, std %.4f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
            print('Cross validation precision mean: %.4f, std %.4f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
            print('Cross validation recall mean: %.4f, std %.4f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
            print('Cross validation fscore mean: %.4f, std %.4f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))

            if args.test_type != 'main' or args.model_ema:
                print('Cross validation accuracy mean: %.4f, std %.4f ' % (np.mean(np.array(acc_ema)), np.std(np.array(acc_ema))))
                print('Cross validation auc mean: %.4f, std %.4f ' % (np.mean(np.array(auc_ema)), np.std(np.array(auc_ema))))

def one_fold(args,device,ckc_metric,te_ckc_metric,ckc_metric_ema,dataset):
    # --->initiation
    if args.random_fold:
        args.seed = args.seed_ori + args.fold_curr*100
    utils.seed_torch(args.seed+args.rank)
    if args.empty_cuda_cache:
        torch.cuda.empty_cache()
    gc.collect()
    global stop_training
    
    loss_scaler = torch.amp.GradScaler(device=device,init_scale=2**args.amp_scale_index,growth_factor=2,growth_interval=args.amp_growth_interval,enabled=not args.amp_unscale)

    amp_autocast = torch.autocast

    # --->build data loader
    train_loader,val_loader,test_loader = build_dataloader(args,dataset)

    # --->bulid networks
    model,model_others = build_model(args,device,train_loader)

    # --->ema 
    if args.model_ema:
        model_ema = utils.ModelEmaV3(model,decay=args.mm,use_warmup=args.mm_sche)
        if args.torchcompile:
            model_ema = torch.compile(model_ema)
    else:
        model_ema = None

    # --->DDP
    if args.distributed:
        model = DDP(model, device_ids=None if args.load_gpu_later or args.load_gpu_later_train else [device],broadcast_buffers=not args.no_ddp_broad_buf)

        if not args.mil_init and 'model_ema' in model_others:
            if args.torchcompile:
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()}
            else:
                state_dict = model.state_dict()

            model_others['model_ema'].load_state_dict(state_dict)

    # --->compile
    if args.torchcompile:
        model = torch.compile(model,dynamic=True,mode=args.torchcompile_mode)
        allow_ops_in_compiled_graph()

    # --->build criterion,optimizer,scheduler,early-stopping
    criterion,optimizer,scheduler,early_stopping = build_train(args,model,train_loader)

    # --->metric
    best_ckc_metric = [0. for i in range(len(ckc_metric))]
    best_ckc_metric_ema = [0. for i in range(len(ckc_metric))]
    best_ckc_metric_te = [0. for i in range(len(te_ckc_metric))]
    best_ckc_metric_te_ema = [0. for i in range(len(te_ckc_metric))]
    epoch_start,opt_thr,opt_main_ema = args.epoch_start,0,0
    _metric_val_ema = None

    # --->build engine
    train,validate = build_engine(args,device,dataset)

    # --->train loop
    train_time_meter = AverageMeter()
    if args.rank == 0 and args.wandb and args.wandb_watch:
        wandb.watch(model,log_freq=int(args.log_iter / 10))
        print(f'Strart Watching Model')
    # if args.distributed:
    #     dist.barrier()
    try:
        if args.script_mode != 'test':
            for epoch in range(epoch_start, args.num_epoch):
                if args.empty_cuda_cache:
                    torch.cuda.empty_cache()
                gc.collect()
                train_loss,start,end = 0,0,0
                if not args.script_mode == "no_train":
                    if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                        train_loader.sampler.set_epoch(epoch)
                    if hasattr(train_loader.dataset, 'set_epoch'):
                        train_loader.dataset.set_epoch(epoch)

                    train_loss,start,end = train(args,model,model_ema,train_loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,epoch,model_others)
                    train_time_meter.update(end-start)

                if args.script_mode == 'only_train':
                    if args.save_iter > 0 and epoch % args.save_iter == 0:
                        save_pt = {
                                    'model': model.state_dict(),
                                    'epoch': epoch
                                }
                        torch.save(save_pt, os.path.join(args.output_path, 'epoch_{epoch}_model.pt'.format(epoch=epoch)))

                    continue

                _metric_val,stop,test_loss, threshold_optimal,rowd_val = validate(args,model=model,loader=val_loader,device=device,criterion=criterion,amp_autocast=amp_autocast,early_stopping=early_stopping,epoch=epoch,others=model_others)
                if args.test_type != 'main':
                    _metric_val,_metric_val_ema = _metric_val

                # validate ema_model
                if model_ema is not None:
                    if args.random_seed_fold > 0:
                        raise NotImplementedError('random_seed_fold and model_ema are not supported in this time')
                    _metric_val_ema,_, test_loss_ema,_,rowd_ema = validate(args,model=model_ema,loader=val_loader,device=device,criterion=criterion,amp_autocast=amp_autocast,epoch=epoch,others=model_others)
                    if args.wandb and args.rank == 0:
                        rowd_ema = OrderedDict([ ('ema_'+_k,_v) for _k, _v in rowd_ema.items()])
                        if rowd_val is not None:
                            rowd_val.update(rowd_ema)

                # always run test_set in the training
                _te_metric = [0.,0.]
                if args.always_test:
                    _te_metric,_te_test_loss_log,rowd = validate(args,model=model,loader=test_loader,device=device,criterion=criterion,amp_autocast=amp_autocast,status='test',others=model_others)
                    if args.wandb and args.rank == 0:
                        rowd = OrderedDict([ (str(args.fold_curr)+'-fold/te_'+_k,_v) for _k, _v in rowd.items()])
                        wandb.log(rowd)

                    if model_ema is not None:
                        _te_ema_metric,rowd,_te_ema_test_loss_log = validate(args,model=model_ema,loader=test_loader,device=device,criterion=criterion,amp_autocast=amp_autocast,status='test')         
                        if args.wandb and args.rank == 0:
                            rowd = OrderedDict([ (str(args.fold_curr)+'-fold/te_ema_'+_k,_v) for _k, _v in rowd.items()])
                            wandb.log(rowd)

                    if args.test_type != 'main':
                        _te_metric,_te_ema_metric = _te_metric
                    else:
                        _te_ema_metric = None

                    if _te_metric[args.best_metric_index] > best_ckc_metric_te[args.best_metric_index]:
                        best_ckc_metric_te = [_te_metric[0],_te_metric[1]]
                        if args.wandb and args.rank == 0:
                            rowd = OrderedDict([
                                ("best_te_main",_te_metric[0]),
                                ("best_te_sub",_te_metric[1])
                            ])
                            rowd = OrderedDict([ (str(args.fold_curr)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                            wandb.log(rowd)
                    
                    if _te_ema_metric is not None:
                        if _te_ema_metric[args.best_metric_index] > best_ckc_metric_te_ema[args.best_metric_index]:
                            best_ckc_metric_te_ema = [_te_ema_metric[0],_te_ema_metric[1]]
                            if args.wandb and args.rank == 0:
                                rowd = OrderedDict([
                                    ("best_te_ema_main",_te_ema_metric[0]),
                                    ("best_te_ema_sub",_te_ema_metric[1])
                                ])
                                rowd = OrderedDict([ (str(args.fold_curr)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                                wandb.log(rowd)
                
                # logging and wandb
                if not args.no_log and args.rank == 0:
                    if args.datasets.lower().startswith('surv'):
                        print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, c-index: %.4f, time: %.4f(%.4f)' % (epoch+1, args.num_epoch, train_loss, test_loss, _metric_val[0], train_time_meter.val,train_time_meter.avg))
                    else:
                        print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.4f, auc_value:%.4f, precision: %.4f, recall: %.4f, fscore: %.4f , time: %.4f(%.4f)' % (epoch+1, args.num_epoch, train_loss, test_loss, _metric_val[1], _metric_val[0], _metric_val[2], _metric_val[3], _metric_val[4], train_time_meter.val,train_time_meter.avg))
                rowd_val['epoch'] = epoch
                if args.wandb and args.rank == 0:
                    if args.random_seed_fold > 0:
                        rowd = OrderedDict([ (f"s{args.rsf_curr}-{args.fold_curr}-fold/val_{_k}",_v) for _k, _v in rowd_val.items()])
                    else:    
                        rowd = OrderedDict([ (str(args.fold_curr)+'-fold/val_'+_k,_v) for _k, _v in rowd_val.items()])
                    wandb.log(rowd)
                
                # update best metric
                if epoch == 0:
                    best_rowd = rowd_val
                else:
                    best_rowd = utils.update_best_metric(best_rowd,rowd_val)

                # save the best model in the val_set
                if _metric_val[args.best_metric_index] > best_ckc_metric[args.best_metric_index]:
                    best_ckc_metric = _metric_val+[epoch]
                    best_rowd['epoch'] = epoch
                    opt_thr = threshold_optimal
                    if not os.path.exists(args.output_path):
                        os.mkdir(args.output_path)
                    if not args.no_log and args.rank==0:
                        _ema_ckp = model_ema.state_dict() if model_ema is not None else None
                        if 'model_ema' in model_others and model_others['model_ema'] is not None:
                            _ema_ckp = model_others['model_ema'].state_dict()
                        best_pt = {
                            'model': model.state_dict(),
                            'teacher': _ema_ckp,
                            'epoch': epoch
                        }
                        torch.save(best_pt, os.path.join(args.output_path, 'fold_{fold}_model_best.pt'.format(fold=args.fold_curr)))

                # save the best model_ema in the val_set
                if _metric_val_ema is not None:
                    if _metric_val_ema[args.best_metric_index] > best_ckc_metric_ema[args.best_metric_index]:
                        best_ckc_metric_ema = _metric_val_ema+[epoch]
                        if not os.path.exists(args.output_path):
                            os.mkdir(args.output_path)
                        if not args.no_log and args.rank==0:
                            _ema_ckp = model_ema.state_dict() if model_ema is not None else None
                            if 'model_ema' in model_others and model_others['model_ema'] is not None:
                                _ema_ckp = model_others['model_ema'].state_dict()
                            best_pt = {
                                'teacher': _ema_ckp,
                                'epoch': epoch
                            }
                            torch.save(best_pt, os.path.join(args.output_path, 'fold_{fold}_ema_model_best.pt'.format(fold=args.fold_curr)))


                if args.wandb and args.rank == 0:
                    if args.random_seed_fold > 0:
                        rowd = OrderedDict([ (f"s{args.rsf_curr}-{args.fold_curr}-fold/val_best_{_k}",_v) for _k, _v in best_rowd.items()])
                    else:
                        rowd = OrderedDict([ (str(args.fold_curr)+'-fold/val_best_'+_k,_v) for _k, _v in best_rowd.items()])
                    wandb.log(rowd)
                
                # save checkpoint
                utils.save_cpk(args,model,random,train_loader,scheduler,optimizer,epoch,early_stopping,_metric_val,_te_metric,best_ckc_metric,best_ckc_metric_te,best_ckc_metric_te_ema,wandb)

                if args.save_iter > 0 and epoch % args.save_iter == 0:
                    save_pt = {
                                'model': model.state_dict(),
                                'epoch': epoch
                            }
                    torch.save(save_pt, os.path.join(args.output_path, 'epoch_{epoch}_model.pt'.format(epoch=epoch)))

                # if args.distributed:
                #     dist.barrier()

                if stop:
                    break

    except KeyboardInterrupt:
        pass
    
    if args.distributed:
        dist.barrier()

    if not args.no_log:
        best_std = None
        if os.path.exists(os.path.join(args.output_path, f'fold_{args.fold_curr}_model_best.pt')):
            best_std = torch.load(os.path.join(args.output_path, 'fold_{fold}_model_best.pt'.format(fold=args.fold_curr)),map_location='cpu',weights_only=True)

        if os.path.exists(os.path.join(args.output_path, f'fold_{args.fold_curr}_ema_model_best.pt')):
            best_std_ema = torch.load(os.path.join(args.output_path, 'fold_{fold}_ema_model_best.pt'.format(fold=args.fold_curr)),map_location='cpu',weights_only=True)

        if best_std is not None:
            info = model.load_state_dict(best_std['model'])
            if args.rank == 0:
                print(f"Epoch {best_std['epoch']} Main Model Loaded: {info}")

        if model_ema is not None and best_std_ema['teacher'] is not None:
            info = model_ema.load_state_dict(best_std_ema['teacher'])
            if args.rank == 0:
                print(f"Epoch {best_std_ema['epoch']} EMA Model Loaded: {info}")
        if args.test_type != 'main':
            info = model_others['model_ema'].load_state_dict(best_std_ema['teacher'])
            if args.rank == 0:
                print(f"Epoch {best_std_ema['epoch']} EMA Model Loaded: {info}")

    metric_test,test_loss_log,rowd = validate(args,model=model,loader=test_loader,device=device,criterion=criterion,amp_autocast=amp_autocast,status='test',others=model_others)

    if model_ema is not None:
        metric_test_ema,test_loss_log,rowd_ema = validate(args,model=model_ema,loader=test_loader,device=device,criterion=criterion,amp_autocast=amp_autocast,status='test',others=model_others)
        if args.wandb and args.rank == 0:
            rowd_ema = OrderedDict([ ('ema_'+_k,_v) for _k, _v in rowd_ema.items()])
            rowd.update(rowd_ema)

    if args.wandb and args.rank == 0:
        if args.random_seed_fold > 0:
            rowd = OrderedDict([ (f"s{args.rsf_curr}/test_{_k}",_v) for _k, _v in rowd.items()])
        else:
            rowd = OrderedDict([ ('test_'+_k,_v) for _k, _v in rowd.items()])
        wandb.log(rowd)
    
    # if args.distributed:
    #     dist.barrier()

    if args.test_type != 'main':
        metric_test,metric_test_ema = metric_test
        [ckc_metric_ema[i].append(metric_test_ema[i]) for i,_ in enumerate(ckc_metric_ema)]
    elif model_ema is not None:
        [ckc_metric_ema[i].append(metric_test_ema[i]) for i,_ in enumerate(ckc_metric_ema)]
    
    # update metric for each-fold
    [ckc_metric[i].append(metric_test[i]) for i,_ in enumerate(ckc_metric)]

    if args.always_test:
        [te_ckc_metric[i].append(best_ckc_metric_te[i]) for i,_ in enumerate(te_ckc_metric)]
        
    return ckc_metric,te_ckc_metric,ckc_metric_ema

if __name__ == '__main__':
    args, args_text = _parse_args()
    args,device = more_about_config(args)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        if args.no_determ:
            # unstable, but fast
            torch.backends.cudnn.benchmark = True
        else:
            #stable, always reproduce results
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if not args.no_deter_algo:
                # torch.use_deterministic_algorithms(True)
                torch.use_deterministic_algorithms(True, warn_only=True)
                torch.utils.deterministic.fill_uninitialized_memory=True
    if 'WANDB_MODE' in os.environ:
            if os.environ["WANDB_MODE"] == "offline":
                _output_wandb_cache = os.path.join(args.output_path,'wandb_cache',args.project,args.title)

    if not os.path.exists(os.path.join(args.output_path,args.project)) and args.rank == 0:
        os.mkdir(os.path.join(args.output_path,args.project))
    args.output_path = os.path.join(args.output_path,args.project,args.title)
    if not os.path.exists(args.output_path) and args.rank == 0:
        os.mkdir(args.output_path)

    #UNI
    if 'UNI_CKPT_PATH' not in os.environ or not os.environ['UNI_CKPT_PATH']:
        os.environ['UNI_CKPT_PATH'] = 'xxx/dataset/wsi_data/ckp/uni/pytorch_model.bin'

    if args.wandb and args.rank == 0:
        utils.check_and_commit_changes(args)

        _output = args.output_path
        
        wandb.init(project=args.project, name=args.title,config=args,dir=os.path.join(_output),tags=["unrecorded"])

        # Get the project root directory
        args.project_root = os.path.dirname(os.path.abspath(__file__))

        # Use os.path.join to create absolute paths
        wandb.save(os.path.join(args.project_root, 'main.py'),base_path=args.project_root, policy='now')
        wandb.save(os.path.join(args.project_root, 'options.py'),base_path=args.project_root, policy='now')
        wandb.save(os.path.join(args.project_root, 'engines/base_engine.py'),base_path=args.project_root, policy='now')
        wandb.save(os.path.join(args.project_root, 'modules/__init__.py'),base_path=args.project_root, policy='now')
        wandb.save(os.path.join(args.project_root, 'datasets/data_utils.py'),base_path=args.project_root, policy='now')
        wandb.save(os.path.join(args.project_root, 'modules/vit_mil.py'),base_path=args.project_root, policy='now')
        
        if args.config:
            for _config in args.config:
                wandb.save(os.path.join(args.project_root, f'config/{_config}'), base_path=args.project_root,policy='now')
  
        if args.pack_bs:
            wandb.save(os.path.join(args.project_root, 'modules/pack.py'), base_path=args.project_root,policy='now')
            wandb.save(os.path.join(args.project_root, 'modules/pack_baseline.py'), base_path=args.project_root,policy='now')
    
    localtime = time.asctime( time.localtime(time.time()) )
    if args.rank == 0:
        print(localtime)
    try:
        if args.random_seed_fold > 0:
            _output = args.output_path
            if args.datasets.lower().startswith('surv'):
                cindex_rsf,cindex_std_rsf,cindex_ema_rsf = [],[],[]
                for i in range(args.random_seed_fold):
                    args.seed = args.seed_ori + i*100
                    args.output_path = os.path.join(_output,f's{i}')
                    args.rsf_curr = i
                    if not args.no_log and args.rank == 0:
                        print('Start %d-fold RSF validation: random %d ' % (args.random_seed_fold, i))
                    cindex,cindex_std, cindex_ema = main(args=args,device=device)
                    if args.wandb and args.rank == 0:
                        wandb.log({
                            f"s{i}/cross_val/cindex_mean":np.mean(np.array(cindex)),
                            f"s{i}/cross_val/cindex_std_mean":np.mean(np.array(cindex_std)),
                            f"s{i}/cross_val/cindex_std":np.std(np.array(cindex)),
                        })
                    cindex_rsf.append(np.mean(np.array(cindex)))
                    cindex_std_rsf.append(np.mean(np.array(cindex_std)))
                    cindex_ema_rsf.append(np.mean(np.array(cindex_ema)))
                if args.wandb and args.rank == 0:
                    wandb.log({
                        "cross_val/cindex_mean":np.mean(np.array(cindex_rsf)),
                        "cross_val/cindex_std_mean":np.mean(np.array(cindex_std_rsf)),
                        "cross_val/cindex_std":np.std(np.array(cindex_rsf)),
                    })
                if args.rank == 0:
                    print('Cross validation c-index mean: %.4f, std %.4f ' % (np.mean(np.array(cindex_rsf)), np.std(np.array(cindex_rsf))))
            else:
                acc_rsf, auc_rsf, fs_rsf, pre_rsf, rec_rsf, ck_rsf, acc_m_rsf = [], [], [], [], [], [], []
                acc_std_rsf, auc_std_rsf, fs_std_rsf = [], [], []
                acc_ema_rsf, auc_ema_rsf, fs_ema_rsf, pre_ema_rsf, rec_ema_rsf, ck_ema_rsf, acc_m_ema_rsf = [], [], [], [], [], [], []
                acc_ema_std_rsf, auc_ema_std_rsf, fs_ema_std_rsf, acc_m_ema_std_rsf = [], [], [], []

                for i in range(args.random_seed_fold):
                    args.seed = args.seed_ori + i * 100
                    args.output_path = os.path.join(_output, f's{i}')
                    args.rsf_curr = i
                    if not args.no_log and args.rank == 0:
                        print('Start %d-fold RSF validation: random %d ' % (args.random_seed_fold, i))

                    (acc, auc, fs, pre, rec, ck, acc_m,
                     acc_std, auc_std, fs_std,
                     acc_ema, auc_ema, fs_ema, pre_ema, rec_ema, ck_ema, acc_m_ema,
                     acc_ema_std, auc_ema_std, fs_ema_std, acc_m_ema_std) = main(args=args, device=device)

                    if args.wandb and args.rank == 0:
                        wandb.log({
                            f"s{i}/cross_val/acc_mean": np.mean(np.array(acc)),
                            f"s{i}/cross_val/auc_mean": np.mean(np.array(auc)),
                            f"s{i}/cross_val/f1_mean": np.mean(np.array(fs)),
                            f"s{i}/cross_val/acc_std_mean": np.mean(np.array(acc_std)),
                            f"s{i}/cross_val/auc_std_mean": np.mean(np.array(auc_std)),
                            f"s{i}/cross_val/f1_std_mean": np.mean(np.array(fs_std)),
                            f"s{i}/cross_val/pre_mean": np.mean(np.array(pre)),
                            f"s{i}/cross_val/recall_mean": np.mean(np.array(rec)),
                            f"s{i}/cross_val/ck_mean": np.mean(np.array(ck)),
                            f"s{i}/cross_val/acc_micro_mean": np.mean(np.array(acc_m)),
                            f"s{i}/cross_val/acc_std": np.std(np.array(acc)),
                            f"s{i}/cross_val/auc_std": np.std(np.array(auc)),
                            f"s{i}/cross_val/f1_std": np.std(np.array(fs)),

                            f"s{i}/cross_val/acc_ema_mean": np.mean(np.array(acc_ema)),
                            f"s{i}/cross_val/auc_ema_mean": np.mean(np.array(auc_ema)),
                            f"s{i}/cross_val/f1_ema_mean": np.mean(np.array(fs_ema)),
                            f"s{i}/cross_val/pre_ema_mean": np.mean(np.array(pre_ema)),
                            f"s{i}/cross_val/recall_ema_mean": np.mean(np.array(rec_ema)),
                            f"s{i}/cross_val/ck_ema_mean": np.mean(np.array(ck_ema)),
                            f"s{i}/cross_val/acc_micro_ema_mean": np.mean(np.array(acc_m_ema)),
                            f"s{i}/cross_val/acc_ema_std_mean": np.mean(np.array(acc_ema_std)),
                            f"s{i}/cross_val/auc_ema_std_mean": np.mean(np.array(auc_ema_std)),
                            f"s{i}/cross_val/f1_ema_std_mean": np.mean(np.array(fs_ema_std)),
                            f"s{i}/cross_val/acc_m_ema_std_mean": np.mean(np.array(acc_m_ema_std)),
                            f"s{i}/cross_val/acc_ema_std": np.std(np.array(acc_ema)),
                            f"s{i}/cross_val/auc_ema_std": np.std(np.array(auc_ema)),
                            f"s{i}/cross_val/f1_ema_std": np.std(np.array(fs_ema)),
                        })

                    acc_rsf.append(np.mean(np.array(acc)))
                    auc_rsf.append(np.mean(np.array(auc)))
                    fs_rsf.append(np.mean(np.array(fs)))
                    pre_rsf.append(np.mean(np.array(pre)))
                    rec_rsf.append(np.mean(np.array(rec)))
                    ck_rsf.append(np.mean(np.array(ck)))
                    acc_m_rsf.append(np.mean(np.array(acc_m)))

                    acc_std_rsf.append(np.mean(np.array(acc_std)))
                    auc_std_rsf.append(np.mean(np.array(auc_std)))
                    fs_std_rsf.append(np.mean(np.array(fs_std)))

                    acc_ema_rsf.append(np.mean(np.array(acc_ema)))
                    auc_ema_rsf.append(np.mean(np.array(auc_ema)))
                    fs_ema_rsf.append(np.mean(np.array(fs_ema)))
                    pre_ema_rsf.append(np.mean(np.array(pre_ema)))
                    rec_ema_rsf.append(np.mean(np.array(rec_ema)))
                    ck_ema_rsf.append(np.mean(np.array(ck_ema)))
                    acc_m_ema_rsf.append(np.mean(np.array(acc_m_ema)))

                    acc_ema_std_rsf.append(np.mean(np.array(acc_ema_std)))
                    auc_ema_std_rsf.append(np.mean(np.array(auc_ema_std)))
                    fs_ema_std_rsf.append(np.mean(np.array(fs_ema_std)))
                    acc_m_ema_std_rsf.append(np.mean(np.array(acc_m_ema_std)))

                if args.wandb and args.rank == 0:
                    wandb.log({
                        "cross_val/acc_mean": np.mean(np.array(acc_rsf)),
                        "cross_val/auc_mean": np.mean(np.array(auc_rsf)),
                        "cross_val/f1_mean": np.mean(np.array(fs_rsf)),
                        "cross_val/acc_std_mean": np.mean(np.array(acc_std_rsf)),
                        "cross_val/auc_std_mean": np.mean(np.array(auc_std_rsf)),
                        "cross_val/f1_std_mean": np.mean(np.array(fs_std_rsf)),
                        "cross_val/pre_mean": np.mean(np.array(pre_rsf)),
                        "cross_val/recall_mean": np.mean(np.array(rec_rsf)),
                        "cross_val/ck_mean": np.mean(np.array(ck_rsf)),
                        "cross_val/acc_micro_mean": np.mean(np.array(acc_m_rsf)),
                        "cross_val/acc_std": np.std(np.array(acc_rsf)),
                        "cross_val/auc_std": np.std(np.array(auc_rsf)),
                        "cross_val/f1_std": np.std(np.array(fs_rsf)),
                    })
                    wandb.log({
                        "cross_val/acc_ema_mean": np.mean(np.array(acc_ema_rsf)),
                        "cross_val/auc_ema_mean": np.mean(np.array(auc_ema_rsf)),
                        "cross_val/f1_ema_mean": np.mean(np.array(fs_ema_rsf)),
                        "cross_val/pre_ema_mean": np.mean(np.array(pre_ema_rsf)),
                        "cross_val/recall_ema_mean": np.mean(np.array(rec_ema_rsf)),
                        "cross_val/ck_ema_mean": np.mean(np.array(ck_ema_rsf)),
                        "cross_val/acc_micro_ema_mean": np.mean(np.array(acc_m_ema_rsf)),
                        "cross_val/acc_ema_std_mean": np.mean(np.array(acc_ema_std_rsf)),
                        "cross_val/auc_ema_std_mean": np.mean(np.array(auc_ema_std_rsf)),
                        "cross_val/f1_ema_std_mean": np.mean(np.array(fs_ema_std_rsf)),
                        "cross_val/acc_m_ema_std_mean": np.mean(np.array(acc_m_ema_std_rsf)),
                        "cross_val/acc_ema_std": np.std(np.array(acc_ema_rsf)),
                        "cross_val/auc_ema_std": np.std(np.array(auc_ema_rsf)),
                        "cross_val/f1_ema_std": np.std(np.array(fs_ema_rsf)),
                    })
    
                if args.rank == 0:
                    print('Cross validation accuracy mean: %.4f, std %.4f ' %
                          (np.mean(np.array(acc_rsf)), np.std(np.array(acc_rsf))))
                    print('Cross validation auc mean: %.4f, std %.4f ' %
                          (np.mean(np.array(auc_rsf)), np.std(np.array(auc_rsf))))
                    print('Cross validation precision mean: %.4f, std %.4f ' %
                          (np.mean(np.array(pre_rsf)), np.std(np.array(pre_rsf))))
                    print('Cross validation recall mean: %.4f, std %.4f ' %
                          (np.mean(np.array(rec_rsf)), np.std(np.array(rec_rsf))))
                    print('Cross validation f1 score mean: %.4f, std %.4f ' %
                          (np.mean(np.array(fs_rsf)), np.std(np.array(fs_rsf))))
            
        else:
            main(args=args,device=device)
    except:
        traceback.print_exc()
    finally:
        pass