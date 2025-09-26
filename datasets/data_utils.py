from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch
import torch.nn.functional as F
import h5py
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
from functools import partial

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def check_tensor(tensor, tensor_name=""):
	if torch.isnan(tensor).any():
		print(f"{tensor_name} contains NaN values")
		raise ValueError
	if torch.isinf(tensor).any():
		print(f"{tensor_name} contains Inf values")
		raise ValueError
	if torch.isfinite(tensor).all():
		pass

def get_seq_pos_fn(h5_path):
	h5_file = h5py.File(h5_path)

	img_x,img_y = h5_file['coords'].attrs['level_dim'] * h5_file['coords'].attrs['downsample']
	patch_size = h5_file['coords'].attrs['patch_size'] * h5_file['coords'].attrs['downsample']
	img_x,img_y = int(img_x),int(img_y)
	patch_size = [int(patch_size[0]),int(patch_size[1])]
	pos = []
	pw = img_x // patch_size[0]
	ph = img_y // patch_size[1]

	for _coord in np.array(h5_file['coords']):
		patch_x = _coord[0] // patch_size[0]  # 水平位置
		patch_y = _coord[1] // patch_size[1]  # 垂直位置

		assert patch_x >= 0 and patch_y >= 0
		if patch_x >= pw:
			pw += 1
		if patch_y >= ph:
			ph += 1

		assert patch_x < pw and patch_y < ph

		pos.append([patch_x,patch_y])

	pos = torch.tensor(np.array(pos,dtype=np.int64))
	pos_all = torch.tensor(np.array([[pw,ph]],dtype=np.int64))

	try:
		check_tensor(pos)
		check_tensor(pos_all)
	except:
		with open('../tmp/log', 'a') as f:
			[print(_pos,file=f) for _pos in pos]
		print(h5_path)
		print(img_x,img_y)
		print(patch_size)
		print(pos)
		print(pos_all)
		assert 1 == 2

	return [pos_all,pos]

def split_by_key(input_list):
	result = defaultdict(list)
	for item in input_list:
		key, sub = item.rsplit('-', 1)
		result[key].append(item)
	return dict(result)

def set_worker_sharing_strategy(worker_id: int) -> None:
	torch.multiprocessing.set_sharing_strategy("file_system")

def get_token_dropout(patch,ratio,min_ps,max_ps):
	if ratio > 0 and ratio < 1:
		keep_size = ratio * len(patch)

	keep_size = int(max(keep_size,min_ps))
	keep_size = int(min(keep_size,max_ps))

	ps = patch.size(0)
	if ps > keep_size:
		idx = torch.randperm(ps)
		patch = patch[idx.long()]
		patch = patch[:int(keep_size)]

	return patch

def get_same_psize(patch,same_psize,_type='zero',min_seq_len=128,pos=None):
	ps = int(patch.size(0))

	if same_psize > 0 and same_psize < 1:
		same_psize = int(same_psize * ps)
		same_psize = max(min_seq_len,same_psize)

	same_psize = int(same_psize)

	if ps < same_psize:
		if _type == 'zero':
			patch = torch.cat([patch,torch.zeros((int(same_psize-ps),patch.size(1)))],dim=0)
			if pos is not None:
				pos = F.pad(pos, (0, 0, 0, int(same_psize-ps)), mode='constant', value=-1)
		elif _type == 'random':
			ori_indices = torch.arange(ps)
			selected_indices = torch.cat([ori_indices, torch.randint(0, ps, (int(same_psize - ps),))])
			patch = patch[selected_indices]
			if pos is not None:
				pos = pos[selected_indices]
		elif _type == 'none':
			pass
		else:
			raise NotImplementedError
	elif ps > same_psize:
		idx = torch.randperm(ps)
		patch = patch[idx.long()]
		patch = patch[:int(same_psize)]
		if pos is not None:
			pos = pos[idx][:same_psize]

	if pos is not None:
		return patch,pos
	
	return patch

def parse_dataframe(args,dataset):
	if args.all_test:
		test_df = dataset['test']
		return None,None,test_df['ID'].tolist(),test_df['Label'].tolist(),None,None
	else:
		train_df = dataset['train']
		test_df = dataset['test']
		val_df = dataset['val']
		return train_df['ID'].tolist(),train_df['Label'].tolist(),test_df['ID'].tolist(),test_df['Label'].tolist(),val_df['ID'].tolist(),val_df['Label'].tolist()

def get_split_dfs(args,df):
	if 'Split' not in df.columns:
		raise ValueError("CSV file must contain a 'Split' column")

	train_df = df[df['Split'].str.lower() == 'train'].reset_index(drop=True)
	test_df = df[df['Split'].str.lower() == 'test'].reset_index(drop=True)
	val_df = df[df['Split'].str.lower() == 'val'].reset_index(drop=True)

	if args.val2test:
		test_df = pd.concat([val_df, test_df], axis=0).reset_index(drop=True)
		args.val_ratio = 0.

	if len(val_df) == 0:
		val_df = test_df

	if args.all_test:
		test_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)
		args.val_ratio = 0.
		train_df = val_df = test_df

	return train_df, test_df, val_df

def get_data_dfs(args, csv_file):
	if args.rank == 0:
		print(f'[dataset] loading dataset from {csv_file}')

	df = pd.read_csv(csv_file)

	required_columns = ['ID', 'Split', 'Label']

	if args.datasets.lower().startswith('surv') and 'Label' not in df.columns:
		df = survival_label(df)

	if not all(col in df.columns for col in required_columns):
		if len(df.columns) == 2:
			from sklearn.model_selection import train_test_split
			df.columns = ['ID', 'Label']
			if args.rank == 0:
				print(f"[dataset] Split column not found in CSV file, splitting data randomly with val_ratio={args.val_ratio}")
			train_indices, test_indices = train_test_split(
				range(len(df)),
				test_size=args.val_ratio,  
				random_state=args.seed  
			)
			df['Split'] = 'train' 
			df.loc[test_indices, 'Split'] = 'test'  
			args.val_ratio = 0.
		elif len(df.columns) == 4:
			df.columns = ['Case', 'ID', 'Label', 'Split']
		else:
			raise ValueError(f"CSV file must contain these columns: {required_columns}")

	if args.rank == 0:
		print(f"Dataset statistics:")
		print(f"Total samples: {len(df)}")
		print(f"Label distribution:")
		print(df['Label'].value_counts())
		print("Split distribution:")
		print(df['Split'].value_counts())

	return df

def get_patient_label(args,csv_file):
	df = pd.read_csv(csv_file)
	required_columns = ['ID','Label']

	if not all(col in df.columns for col in required_columns):
		if len(df.columns) == 2:
			df.columns = ['ID', 'Label']
		else:
			df.columns = ['Case', 'ID', 'Label', 'Split']

	patients_list = df['ID']
	labels_list = df['Label']

	label_counts = labels_list.value_counts().to_dict()

	if args:
		if args.rank == 0:
			print(f"patient_len:{len(patients_list)} label_len:{len(labels_list)}")
			print(f"all_counter:{label_counts}")

	return df

def get_patient_label_surv(args,csv_file):
	if args:
		if args.rank == 0:
			print('[dataset] loading dataset from %s' % (csv_file))
	rows = pd.read_csv(csv_file)
	rows = survival_label(rows)

	label_dist = rows['Label'].value_counts().sort_index()

	if args:
		if args.rank == 0:
			print('[dataset] discrete label distribution: ')
			print(label_dist)
			print('[dataset] dataset from %s, number of cases=%d' % (csv_file, len(rows)))

	return rows

def data_split(seed,df, ratio, shuffle=True, label_balance_val=True):
	if label_balance_val:
		val_df = pd.DataFrame()
		train_df = pd.DataFrame()

		for label in df['Label'].unique():
			label_df = df[df['Label'] == label]
			n_total = len(label_df)
			offset = int(n_total * ratio)

			if shuffle:
				label_df = label_df.sample(frac=1, random_state=seed)

			val_df = pd.concat([val_df, label_df.iloc[:offset]])
			train_df = pd.concat([train_df, label_df.iloc[offset:]])
	else:
		n_total = len(df)
		offset = int(n_total * ratio)

		if n_total == 0 or offset < 1:
			return pd.DataFrame(), df

		if shuffle:
			df = df.sample(frac=1, random_state=seed)

		val_df = df.iloc[:offset]
		train_df = df.iloc[offset:]

	return val_df, train_df

def get_kfold(args,k, df, val_ratio=0, label_balance_val=True):
	if k <= 1:
		raise NotImplementedError

	skf = StratifiedKFold(n_splits=k)

	train_dfs = []
	test_dfs = []
	val_dfs = []

	for train_index, test_index in skf.split(df, df['Label']):
		train_df = df.iloc[train_index]
		test_df = df.iloc[test_index]

		if val_ratio != 0:
			val_df, train_df = data_split(args.seed,train_df, val_ratio, True, label_balance_val)

			if args.val2test:
				test_df = pd.concat([val_df, test_df], axis=0).reset_index(drop=True)
				args.val_ratio = 0.
		else:
			val_df = pd.DataFrame()

		train_dfs.append(train_df)
		test_dfs.append(test_df)
		val_dfs.append(val_df)

	return train_dfs, test_dfs, val_dfs

def survival_label(rows):
	n_bins, eps = 4, 1e-6
	uncensored_df = rows[rows['Status'] == 1]
	disc_labels, q_bins = pd.qcut(uncensored_df['Event'], q=n_bins, retbins=True, labels=False)
	q_bins[-1] = rows['Event'].max() + eps
	q_bins[0] = rows['Event'].min() - eps
	disc_labels, q_bins = pd.cut(rows['Event'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
	# missing event data
	disc_labels = disc_labels.values.astype(int)
	disc_labels[disc_labels < 0] = -1
	if 'Label' not in rows.columns:
		rows.insert(len(rows.columns), 'Label', disc_labels)
	# Remove rows with label -1
	rows = rows[rows['Label'] != -1].reset_index(drop=True)
	return rows

def collate_fn_nbs(batch):
	return batch[0]

def collate_fn_pack(batch):
	inputs = [item['input'] for item in batch]
	targets = torch.tensor([item['target'] for item in batch], dtype=torch.long) 

	result = {
		'input': inputs,
		'target': targets
	}

	if any('event' in item for item in batch):
		result['event'] = torch.tensor([item['event'] for item in batch], dtype=torch.float32)
	if any('censorship' in item for item in batch):
		result['censorship'] = torch.tensor([item['censorship'] for item in batch], dtype=torch.float32)
	if any('pos' in item for item in batch): 
		result['pos'] = [item['pos'] for item in batch if 'pos' in item]
	if any('idx' in item for item in batch):
		result['idx'] = [item['idx'] for item in batch if 'idx' in item]

	return result

class PrefetchLoader:
	def __init__(
			self,
			loader,
			mean=IMAGENET_MEAN,
			std=IMAGENET_STD,
			channels=3,
			device=torch.device('cuda'),
			img_dtype=torch.float32,
			need_norm=True,
			need_transform=False,
			transform_type='strong',
			img_size=224,
			is_train=True,
			trans_chunk=4,
			crop_scale=0.08,
			load_gpu_later=False):

		normalization_shape = (1, channels, 1, 1)

		self.loader = loader
		self.need_norm = need_norm

		if need_transform:
			if is_train:
				# ta_list = [
				# 		ta_transforms.RandomHorizontalFlip(p=0.5),
				# 		ta_transforms.RandomVerticalFlip(p=0.5),
				# 		#ta_transforms.ColorJitter(0.2, 0., 0., 0.),
				# 		#ta_transforms.RandomResizedCrop(224),
				# 		#ta_transforms.RandomCrop(224),
				# 	]  if transform_type == 'strong' else[
				# 		# ta_transforms.RandomHorizontalFlip(p=0.5),
				# 		# ta_transforms.RandomVerticalFlip(p=0.5),
				# 		#ta_transforms.ColorJitter(0.2, 0., 0., 0.),
				# 		#ta_transforms.RandomResizedCrop(224),
				# 		#ta_transforms.RandomCrop(224),
				# 	]
				if transform_type == 'strong':
					ta_list = [
						ta_transforms.RandomHorizontalFlip(p=0.5),
						ta_transforms.RandomVerticalFlip(p=0.5),
						# ta_transforms.ColorJitter(0.2, 0., 0., 0.),
						# ta_transforms.RandomResizedCrop(224),
						# ta_transforms.RandomCrop(224),
					]
				elif transform_type == 'strong_v2':
					ta_list = [
						ta_transforms.RandomHorizontalFlip(p=0.5),
						ta_transforms.RandomVerticalFlip(p=0.5),
						ta_transforms.RandomAffine(degrees=0, translate=(0.5, 0.5)),
					]
				elif transform_type == 'weak_strong':
					ta_list = [
						# ta_transforms.RandomHorizontalFlip(p=0.5),
						# ta_transforms.RandomAffine(degrees=0, translate=(0.5, 0.5)),
						ta_transforms.RandomAffine(degrees=0, translate=(100 / 256, 100 / 256)),
						#ta_transforms.RandomRotation(degrees=30),
					]
				else:
					ta_list = [
						# ta_transforms.RandomHorizontalFlip(p=0.5),
						# ta_transforms.RandomVerticalFlip(p=0.5),
						# ta_transforms.ColorJitter(0.2, 0., 0., 0.),
						# ta_transforms.RandomResizedCrop(224),
						# ta_transforms.RandomCrop(224),
					]
				if img_size != 224:
					ta_list += [
						ta_transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
					]
				else:
					ta_list += [
						ta_transforms.Resize(224),
					]
				self.transform = ta_transforms.SequentialTransform(
					ta_list,
					inplace=True,
					batch_inplace=True,
					batch_transform=True,
					num_chunks=trans_chunk,
					permute_chunks=False,
				)
			else:
				self.transform = ta_transforms.SequentialTransform(
					[
						ta_transforms.CenterCrop(224),
					],
					inplace=True,
					batch_inplace=True,
					batch_transform=True,
					num_chunks=1,
					permute_chunks=False,
				)
		else:
			self.transform = None
		self.device = self.device_input = device
		# if fp16:
		# 	# fp16 arg is deprecated, but will override dtype arg if set for bwd compat
		# 	img_dtype = torch.float16
		self.img_dtype = img_dtype
		self.no_gpu_key = []
		if load_gpu_later:
			device = torch.device('cpu')
			self.device_input=device
			self.no_gpu_key = ['input']
		
		self.mean = torch.tensor(
			[x * 255 for x in mean], device=device, dtype=img_dtype).view(normalization_shape)
		self.std = torch.tensor(
			[x * 255 for x in std], device=device, dtype=img_dtype).view(normalization_shape)

		self.is_cuda = torch.cuda.is_available() and device.type == 'cuda'

	def __iter__(self):
		first = True
		if self.is_cuda:
			stream = torch.cuda.Stream()
			stream_context = partial(torch.cuda.stream, stream=stream)
		else:
			stream = None
			stream_context = suppress

		for next_batch in self.loader:
			with stream_context():
				# 将所有张量移动到设备
				next_batch = {
					k: ([v.to(device=self.device_input, non_blocking=True) for v in v] if isinstance(v, list) and k == 'input' else 
						v.to(device=self.device, non_blocking=True) if isinstance(v, torch.Tensor) and k not in self.no_gpu_key else v)
					for k, v in next_batch.items()
				}
				# 对输入进行变换和归一化
				if self.transform is not None:
					if isinstance(next_batch['input'], list):
						next_batch['input'] = [self.transform(tensor) for tensor in next_batch['input']]
					else:
						next_batch['input'] = self.transform(next_batch['input'])
				if self.need_norm:
					if isinstance(next_batch['input'], list):
						next_batch['input'] = [tensor.to(self.img_dtype).sub_(self.mean).div_(self.std) for tensor in next_batch['input']]
					else:
						next_batch['input'] = next_batch['input'].to(self.img_dtype).sub_(self.mean).div_(self.std)

			if not first:
				yield batch
			else:
				first = False

			if stream is not None:
				torch.cuda.current_stream().wait_stream(stream)

			batch = next_batch

		yield batch

	def __len__(self):
		return len(self.loader)
	@property
	def sampler(self):
		return self.loader.sampler

	@property
	def dataset(self):
		return self.loader.dataset