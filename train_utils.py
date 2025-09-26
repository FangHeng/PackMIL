import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.scheduler import create_scheduler_v2
import math

from utils import *
from timm.loss import AsymmetricLossSingleLabel,LabelSmoothingCrossEntropy

def adjust_encoder_learning_rate(model,n_epoch_warmup, n_epoch, max_lr, optimizer, dloader_len, step):
    """
    Set learning rate according to cosine schedule
    """

    max_steps = int(n_epoch * dloader_len)
    warmup_steps = int(n_epoch_warmup * dloader_len)
    step += 1

    if step < warmup_steps:
        lr = max_lr * step / warmup_steps
        #lr = 0.
        # model.freeze_encoder()
    else:
        # model.unfreeze_encoder()
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = max_lr * 0.001
        lr = max_lr * q + end_lr * (1 - q)
        
    #print(optimizer.param_groups[0])
    optimizer.param_groups[0]['lr'] = lr

def zero_learning_rate(optimizer,mode='enc'):
    """
    Set learning rate according to cosine schedule
    """
        
    #print(optimizer.param_groups[0])
    if mode == 'enc':
        optimizer.param_groups[0]['lr'] = 0.
    elif mode == 'mil':
        #print(optimizer.param_groups[1])
        optimizer.param_groups[-1]['lr'] = 0.


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


############# Survival Prediction ###################
def nll_loss(hazards,S,Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1)  # censorship status, 0 or 1
    # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    
    return loss

class NLLSurvLoss(object):
    def __init__(self, alpha=0.):
        self.alpha = alpha

    def __call__(self, Y, c, logits=None,hazards=None, S=None,alpha=None):
        if alpha is None:
            alpha = self.alpha
        if hazards is None:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
        return nll_loss(hazards, S, Y, c, alpha=alpha)

def nll_loss_mixup(hazards,S,Y, c, alpha=0.4, eps=1e-7,censored_enable=True,Y_censored=None):

    batch_size = len(Y)
    Y = Y.view(batch_size, 1).to(torch.int64)  # ground truth bin, 1,2,...,k
    Y_censored = Y_censored.view(batch_size, 1).to(torch.int64)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1)  # censorship status, 0 or 1
    # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y_censored + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()

    return loss

class NLLSurvMulLoss(object):
    def __init__(self, alpha=0.,censored_enable=True):
        self.alpha = alpha
        self.censored_enable=censored_enable
    def __call__(self, Y, c, logits=None,hazards=None, S=None,alpha=None,Y_censored=None):
        if alpha is None:
            alpha = self.alpha
        if hazards is None:
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
        return nll_loss_mixup(hazards, S, Y, c, alpha=alpha,censored_enable=self.censored_enable,Y_censored=Y_censored)

class BCESurvLoss(object):
    def __init__(self,surv_enable=True,censored_enable=True):
        self.surv_enable = surv_enable
        self.censored_enable = censored_enable
    def __call__(self, Y, c, logits=None, Y_censored=None):
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        #batch_size = len(Y)
        n_bin = len(Y[0])
        #_pad_shape = c[:,0].view(batch_size,1)
        #S_padded = torch.cat([torch.ones_like(_pad_shape), S], 1)

        if self.surv_enable:
            surv_loss = F.binary_cross_entropy(
                S[:,:n_bin-1],
                Y[:,1:],
                1-c,
                reduction="mean",
            )
        else:
            surv_loss = 0.
        hazards_loss = F.binary_cross_entropy_with_logits(
            logits,
            Y,
            1-c,
            pos_weight=None,
            reduction="mean",
        )
        uncensored_loss =  surv_loss + hazards_loss

        #print(uncensored_loss)

        if self.censored_enable:
            censored_loss = F.binary_cross_entropy(
                S,
                Y_censored,
                c,
                reduction="mean",
            )
            
        else:
            censored_loss = 0.
        
        #print(censored_loss)

        loss = uncensored_loss + censored_loss
        #print(loss)
        return loss

class cox_loss_v2(nn.Module):
    def __init__(self):  #weight=None, size_average=True):
        super(cox_loss_v2, self).__init__()

    def forward(self, Y, c, logits):
    # def cox_loss(y_true, y_pred): 
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        survtime = Y
        censor = c.bool()
        hazard_pred = torch.sigmoid(logits)

        current_batch_len = len(survtime)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = survtime[j] >= survtime[i]

        R_mat = torch.FloatTensor(R_mat)
        theta = hazard_pred.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
        return loss_cox

def build_train(args,model,train_loader):
    # build criterion
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss(label_smoothing=args.label_smooth)
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    elif args.loss == "nll_surv":
        criterion = NLLSurvLoss(alpha=0.0)
    elif args.loss == 'asl':
        criterion = AsymmetricLossSingleLabel(gamma_neg=4, gamma_pos=1, eps=args.label_smooth)

    # build optimizer
    if args.distributed:
        _model = model.module
    else:
        _model = model

    # lr scale
    if args.lr_scale:
        global_batch_size = args.batch_size * args.world_size * args.accumulation_steps
        batch_ratio = global_batch_size / args.lr_base_size
        batch_ratio = batch_ratio ** 0.5
        #batch_ratio = max(batch_ratio,4)
        lr = args.lr * batch_ratio
    else:
        lr = args.lr

    params = [
        {'params': filter(lambda p: p.requires_grad, _model.parameters()), 'lr': lr,'weight_decay': args.weight_decay},]
            
    
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(params)
    elif args.opt == 'adam':
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(params)

    # build scheduler
    if args.lr_sche == 'cosine':
        scheduler,_ = create_scheduler_v2(optimizer,sched='cosine',num_epochs=args.num_epoch,warmup_lr=args.warmup_lr,warmup_epochs=args.warmup_epochs,min_lr=1e-7)

    elif args.lr_sche == 'step':
        assert not args.lr_supi
        # follow the DTFD-MIL
        # ref:https://github.com/hrzhang1123/DTFD-MIL
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const':
        scheduler = None

    # build early stopping
    if args.early_stopping:
        # if args.datasets.lower().startswith('surv'):
        #     #patience,stop_epoch = 10,args.max_epoch
        # else:
            #patience,stop_epoch = 20,args.max_epoch
        patience,stop_epoch = args.patient,args.max_epoch
        early_stopping = EarlyStopping(patience=patience, stop_epoch=stop_epoch)
    else:
        early_stopping = None
    
    return criterion,optimizer,scheduler,early_stopping