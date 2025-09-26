import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
from torch import distributed as dist
from copy import deepcopy
from typing import Optional
import os
from git import Repo

def seed_torch(seed=2021):
		import random
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		#os.environ['PYTHONHASHSEED'] = str(seed)

def check_and_commit_changes(args):
    if args.rank == 0:
        # Get git repository of current directory
        repo = Repo(os.path.dirname(os.path.abspath(__file__)))
        # Check for uncommitted changes
        has_changes = repo.is_dirty()
        has_untracked = len(repo.untracked_files) > 0

        if has_changes or has_untracked:
            print("Detected uncommitted changes...")
            # Handle untracked files
            if has_untracked:
                print("Untracked files detected...")
                repo.git.add(repo.untracked_files)
                #raise NotImplementedError

            commit_message = "Run Auto commit"
            # Commit all changes
            repo.git.commit('-a', '-m', commit_message)
            print(f"Changes committed: {commit_message}")

            # Push to remote repository
            # print("Pushing to remote repository...")
            # origin = repo.remote('origin')
            # origin.push('master')
            # print("Push completed")
        else:
            print("No changes detected")

class ModelEmaV3(nn.Module):
	def __init__(
			self,
			model,
			decay: float = 0.9999,
			min_decay: float = 0.0,
			update_after_step: int = 0,
			use_warmup: bool = False,
			warmup_gamma: float = 1.0,
			warmup_power: float = 2/3,
			device: Optional[torch.device] = None,
			foreach: bool = True,
			exclude_buffers: bool = False,
			mm_sche=None,
	):
		super().__init__()
		# make a copy of the model for accumulating moving average of weights
		self.module = deepcopy(model)
		self.module.eval()
		self.decay = decay
		self.min_decay = min_decay
		self.update_after_step = update_after_step
		self.mm_sche = mm_sche
		self.use_warmup = use_warmup
		self.warmup_gamma = warmup_gamma
		self.warmup_power = warmup_power
		self.foreach = foreach
		self.device = device  # perform ema on different device from model if set
		self.exclude_buffers = exclude_buffers
		if self.device is not None and device != next(model.parameters()).device:
			self.foreach = False  # cannot use foreach methods with different devices
			self.module.to(device=device)

	def get_decay(self, step: Optional[int] = None) -> float:
		"""
		Compute the decay factor for the exponential moving average.
		"""
		if step is None:
			return self.decay

		step = max(0, step - self.update_after_step - 1)
		# if step <= 0:
		# 	return 0.0
		if step < 0:
			return 0.0

		if self.use_warmup:
			decay = 1 - (1 + step / self.warmup_gamma) ** -self.warmup_power
			decay = max(min(decay, self.decay), self.min_decay)
		else:
			decay = self.decay

		if self.mm_sche:
			decay = self.mm_sche[step]

		return decay

	@torch.no_grad()
	def update(self, model, step: Optional[int] = None):
		if self.decay == 1.:
			return None
		decay = self.get_decay(step) 
		if self.exclude_buffers:
			self.apply_update_no_buffers_(model, decay)
		else:
			self.apply_update_(model, decay)

	def apply_update_(self, model, decay: float):
		# interpolate parameters and buffers
		if self.foreach:
			ema_lerp_values = []
			model_lerp_values = []

			ema_state_dict = self.module.state_dict()
			model_state_dict = model.state_dict()
			for name, ema_v in ema_state_dict.items():
				if name in model_state_dict:
					model_v = model_state_dict[name]
				# ddp
				elif f"module.{name}" in model_state_dict:
					model_v = model_state_dict[f"module.{name}"]
				# torchcompile + ddp
				elif f"_orig_mod.module.{name}" in model_state_dict:
					model_v = model_state_dict[f"_orig_mod.module.{name}"]
				# torchcompile
				elif f"_orig_mod.{name}" in model_state_dict:
					model_v = model_state_dict[f"_orig_mod.{name}"]
				else:
					# print(f"Skipping parameter {name} as it's not found in source model")
					continue

				if ema_v.is_floating_point():
					ema_lerp_values.append(ema_v)
					model_lerp_values.append(model_v)
				else:
					ema_v.copy_(model_v)

			if hasattr(torch, '_foreach_lerp_'):
				torch._foreach_lerp_(ema_lerp_values, model_lerp_values, weight=1. - decay)
			else:
				torch._foreach_mul_(ema_lerp_values, scalar=decay)
				torch._foreach_add_(ema_lerp_values, model_lerp_values, alpha=1. - decay)
		else:
			for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
				if ema_v.is_floating_point():
					ema_v.lerp_(model_v.to(device=self.device), weight=1. - decay)
				else:
					ema_v.copy_(model_v.to(device=self.device))

	def apply_update_no_buffers_(self, model, decay: float):
		# interpolate parameters, copy buffers
		ema_params = tuple(self.module.parameters())
		model_params = tuple(model.parameters())
		if self.foreach:
			if hasattr(torch, '_foreach_lerp_'):
				torch._foreach_lerp_(ema_params, model_params, weight=1. - decay)
			else:
				torch._foreach_mul_(ema_params, scalar=decay)
				torch._foreach_add_(ema_params, model_params, alpha=1 - decay)
		else:
			for ema_p, model_p in zip(ema_params, model_params):
				ema_p.lerp_(model_p.to(device=self.device), weight=1. - decay)

		for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
			ema_b.copy_(model_b.to(device=self.device))

	@torch.no_grad()
	def set(self, model):
		for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
			ema_v.copy_(model_v.to(device=self.device))

	def forward(self, *args, **kwargs):
		return self.module(*args, **kwargs)

def reduce_tensor(tensor, n):

	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.ReduceOp.SUM)
	rt /= n
	return rt

def distributed_concat(tensor,num_sample):
	output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
	dist.all_gather(output_tensors, tensor)
	concat = torch.cat(output_tensors, dim=0)
	return concat[:num_sample]

def patch_shuffle(x,group=0,g_idx=None,return_g_idx=False):
	b,p,n = x.size()
	ps = torch.tensor(list(range(p)))

	# padding
	H, W = int(np.ceil(np.sqrt(p))), int(np.ceil(np.sqrt(p)))
	if group > H or group<= 0:
		return group_shuffle(x,group)
	_n = -H % group
	H, W = H+_n, W+_n
	add_length = H * W - p
	# print(add_length)
	ps = torch.cat([ps,torch.tensor([-1 for i in range(add_length)])])
	# patchify
	ps = ps.reshape(shape=(group,H//group,group,W//group))
	ps = torch.einsum('hpwq->hwpq',ps)
	ps = ps.reshape(shape=(group**2,H//group,W//group))
	# shuffle
	if g_idx is None:
		g_idx = torch.randperm(ps.size(0))
	ps = ps[g_idx]
	# unpatchify
	ps = ps.reshape(shape=(group,group,H//group,W//group))
	ps = torch.einsum('hwpq->hpwq',ps)
	ps = ps.reshape(shape=(H,W))
	idx = ps[ps>=0].view(p)
	
	if return_g_idx:
		return x[:,idx.long()],g_idx
	else:
		return x[:,idx.long()]

def group_shuffle(x,group=0):
	b,p,n = x.size()
	ps = torch.tensor(list(range(p)))
	if group > 0 and group < p:
		_pad = -p % group
		ps = torch.cat([ps,torch.tensor([-1 for i in range(_pad)])])
		ps = ps.view(group,-1)
		g_idx = torch.randperm(ps.size(0))
		ps = ps[g_idx]
		idx = ps[ps>=0].view(p)
	else:
		idx = torch.randperm(p)
	return x[:,idx.long()]

def optimal_thresh(fpr, tpr, thresholds, p=0):
	loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
	idx = np.argmin(loss, axis=0)
	return fpr[idx], tpr[idx], thresholds[idx]

def save_cpk(args,model,random,train_loader,scheduler,optimizer,epoch,early_stopping,_metric_val,_te_metric,best_ckc_metric,best_ckc_metric_te,best_ckc_metric_te_tea,wandb):
	random_state = {
		'np': np.random.get_state(),
		'torch': torch.random.get_rng_state(),
		'py': random.getstate(),
		'loader': '',
	}
	ckp = {
		'model': model.state_dict(),
		'lr_sche': scheduler.state_dict(),
		'optimizer': optimizer.state_dict(),
		'epoch': epoch+1,
		'k': args.fold_curr,
		'early_stop': early_stopping.state_dict(),
		'random': random_state,
		'ckc_metric': _metric_val+_te_metric,
		'val_best_metric': best_ckc_metric,
		'te_best_metric': best_ckc_metric_te+best_ckc_metric_te_tea,
		#'wandb_id': wandb.run.id if args.wandb else '',
	}
	if args.rank == 0:
		torch.save(ckp, os.path.join(args.output_path, 'ckp.pt'))

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))
	labels = np.array(dataset.slide_label)
	label_uni = set(dataset.slide_label)
	weight_per_class = [N/len(labels[labels==c]) for c in label_uni]
	weight = [0] * int(N)
	for idx in range(len(dataset)):
		y = dataset.slide_label[idx]
		weight[idx] = weight_per_class[y]

	return torch.DoubleTensor(weight)

def multi_class_scores(true_labels, pred_probs, classes):
	true_labels_bin = label_binarize(true_labels, classes=classes)

	macro_auc = roc_auc_score(true_labels_bin, pred_probs, average='macro', multi_class='ovr')

	predictions = np.argmax(pred_probs, axis=1)

	precision, recall, fscore, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
	accuracy = accuracy_score(true_labels, predictions)

	return accuracy, macro_auc, precision, recall, fscore

def five_scores(bag_labels, bag_predictions,threshold_optimal=None):
	bag_predictions = bag_predictions[:,1]
	if threshold_optimal is None:
		fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
		fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
	# threshold_optimal=0.5
	auc_value = roc_auc_score(bag_labels, bag_predictions)
	this_class_label = np.array(bag_predictions)
	this_class_label[this_class_label>=threshold_optimal] = 1
	this_class_label[this_class_label<threshold_optimal] = 0
	bag_predictions = this_class_label
	precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
	accuracy = accuracy_score(bag_labels, bag_predictions)
	return accuracy, auc_value, precision, recall, fscore,threshold_optimal

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
	warmup_schedule = np.array([])
	warmup_iters = warmup_epochs * niter_per_ep
	if warmup_epochs > 0:
		warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

	iters = np.arange(epochs * niter_per_ep - warmup_iters)
	schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

	schedule = np.concatenate((warmup_schedule, schedule))
	assert len(schedule) == epochs * niter_per_ep
	return schedule

def update_best_metric(best_metric,val_metric):
	updated_best_metric = best_metric.copy()

	assert set(best_metric.keys()) == set(val_metric.keys()), "两个指标字典的键必须相同"

	for key in val_metric.keys():
		if key == "epoch":
			continue
		if key == "loss":
			if val_metric[key] < best_metric[key]:
				updated_best_metric[key] = val_metric[key]
		else:
			if val_metric[key] > best_metric[key]:
				updated_best_metric[key] = val_metric[key]

	return updated_best_metric

class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, patience=20, stop_epoch=50, verbose=False):
		"""
		Args:
			patience (int): How long to wait after last time validation loss improved.
							Default: 20
			stop_epoch (int): Earliest epoch possible for stopping
			verbose (bool): If True, prints a message for each validation loss improvement. 
							Default: False
		"""
		self.patience = patience
		self.stop_epoch = stop_epoch
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		try:
			self.val_loss_min = np.Inf
		except:
			self.val_loss_min = np.inf

	def __call__(self, args, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):
		
		score = -val_loss
		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model, ckpt_name)
		elif score < self.best_score:
			self.counter += 1
			if args.rank == 0:
				print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience and epoch > self.stop_epoch:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model, ckpt_name)
			self.counter = 0

	def state_dict(self):
		return {
			'patience': self.patience,
			'stop_epoch': self.stop_epoch,
			'verbose': self.verbose,
			'counter': self.counter,
			'best_score': self.best_score,
			'early_stop': self.early_stop,
			'val_loss_min': self.val_loss_min
		}
	def load_state_dict(self,dict):
		self.patience = dict['patience']
		self.stop_epoch = dict['stop_epoch']
		self.verbose = dict['verbose']
		self.counter = dict['counter']
		self.best_score = dict['best_score']
		self.early_stop = dict['early_stop']
		self.val_loss_min = dict['val_loss_min']

	def save_checkpoint(self, val_loss, model, ckpt_name):
		'''Saves model when validation loss decrease.'''
		if self.verbose:
			print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		#torch.save(model.state_dict(), ckpt_name)
		self.val_loss_min = val_loss
