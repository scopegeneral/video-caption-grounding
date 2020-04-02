import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):

	def get_vocab_size(self):
		return len(self.get_vocab())

	def get_vocab(self):
		return self.ix_to_word

	def get_seq_length(self):
		return self.seq_length

	def __init__(self, opts):
		super(VideoDataset, self).__init__()
		

		# load the json file which contains information about the dataset
		self.captions = json.load(open(opts["caption_json"]))
		self.labels = json.load(open(opts["timestamps_json"]))
		info = json.load(open(opts["info_json"]))
		self.ix_to_word = info['ix_to_word']
		self.word_to_ix = info['word_to_ix']
		print('vocab size is ', len(self.ix_to_word))
		self.splits = info['videos']
		print('number of train videos: ', len(self.splits['train']))
		print('number of test videos: ', len(self.splits['test']))

		self.feats_dir = opts["feats_dir"]
		#self.c3d_feats_dir = opts['c3d_feats_dir']
		#self.with_c3d = opts['with_c3d']
		print('load feats from %s' % (self.feats_dir))
		# load in the sequence data
		self.max_len = opts["max_len"]
		print('max sequence length in data is', self.max_len)

	def __getitem__(self, ix):
		"""This function returns a tuple that is further passed to collate_fn
		"""
		# which part of data to load
		#if self.mode == 'test':
			#ix = ix + len(self.splits['train'])
		
		video_feat.append(np.load(os.path.join(dir, 'video%i.npy' % (self.captions[ix]['video']))))
		
		caption_feat = self.captions[ix]['caption']
		if len(caption_feat) > self.max_len:
			caption_feat = caption_feat[:self.max_len]
			caption_feat[-1] = '<eos>'
		"""
		if self.with_c3d == 1:
			c3d_feat = np.load(os.path.join(self.c3d_feats_dir, 'video%i.npy'%(ix)))
			c3d_feat = np.mean(c3d_feat, axis=0, keepdims=True)
			fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)
		"""
		
		mask = np.zeros(self.max_len)
		caption_text = self.captions[ix]['caption']
		caption = np.zeros(self.max_len)
		if len(caption_text) > self.max_len:
			caption_text = caption_text[:self.max_len]
			caption[-1] = '<eos>'
		for j, w in enumerate(caption_text):
			caption[j] = self.word_to_ix[w]
			
		# random select a caption for this video
		label = np.array(self.labels[ix]['label']/self.labels[ix]['duration'])	
		non_zero = (caption == 0).nonzero()
		mask[:int(non_zero[0]) + 1] = 1

		data = {}
		data['video_feat'] = torch.from_numpy(video_feat).type(torch.FloatTensor)
		data['caption_feat'] = torch.from_numpy(caption_feat).type(torch.FloatTensor)
		data['label'] = torch.from_numpy(label).type(torch.FloatTensor)
		data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
		data['duration'] = torch.Tensor([self.labels[ix]['duration']]).type(torch.FloatTensor)
		data['video_id'] = 'video%i'%(self.captions[ix]['video'])
		return data

	def __len__(self, mode):
		return len(self.splits[mode])

def pad_tensor(vec, pad, dim):
	"""
	args:
		vec - tensor to pad
		pad - the size to pad to
		dim - dimension to pad

	return:
		a new tensor padded to 'pad' in dimension 'dim'
	"""
	pad_size = list(vec.shape)
	pad_size[dim] = pad - vec.size(dim)
	return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate:
	"""
	a variant of callate_fn that pads according to the longest sequence in
	a batch of sequences
	"""

	def __init__(self, dim=0):
		"""
		args:
			dim - the dimension to be padded (dimension of time in sequences)
		"""
		self.dim = dim

	def pad_collate(self, batch):
		"""
		args:
			batch - list of (tensor, label)

		reutrn:
			xs - a tensor of all examples in 'batch' after padding
			ys - a LongTensor of all labels in batch
		"""
		# find longest sequence
		max_len = max(map(lambda x: x['video_feat'].shape[self.dim], batch))
		# pad according to max_len
		batch = map(lambda x:
					(pad_tensor(x['video_feat'], pad=max_len, dim=self.dim), y), batch)
		# stack all
		xs = torch.stack(map(lambda x: x['video_feat'], batch), dim=0)
		return xs

	def __call__(self, batch):