# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import pickle as pickle
import numpy as np
import pandas as pd
import random
import os
import glob

import io
import torch
from torchvision.utils import save_image
from PIL import Image

from .utils import pad_seq, bytes_to_file, \
	read_split_image, shift_and_resize_image, normalize_image, \
	tight_crop_image, add_padding

# def get_batch_iter(examples, batch_size, augment):
# 	# the transpose ops requires deterministic
# 	# batch size, thus comes the padding
# 	padded = pad_seq(examples, batch_size)

# 	def process(img):
# 		img = bytes_to_file(img)
# 		try:
# 			img_A, img_B = read_split_image(img)
# 			if augment:
# 				# augment the image by:
# 				# 1) enlarge the image
# 				# 2) random crop the image back to its original size
# 				# NOTE: image A and B needs to be in sync as how much
# 				# to be shifted
# 				w, h, _ = img_A.shape
# 				multiplier = random.uniform(1.00, 1.20)
# 				# add an eps to prevent cropping issue
# 				nw = int(multiplier * w) + 1
# 				nh = int(multiplier * h) + 1
# 				shift_x = int(np.ceil(np.random.uniform(0.01, nw - w)))
# 				shift_y = int(np.ceil(np.random.uniform(0.01, nh - h)))
# 				img_A = shift_and_resize_image(img_A, shift_x, shift_y, nw, nh)
# 				img_B = shift_and_resize_image(img_B, shift_x, shift_y, nw, nh)
# 			img_A = normalize_image(img_A)
# 			img_B = normalize_image(img_B)
# 			return np.concatenate([img_A, img_B], axis=2)
# 		finally:
# 			img.close()

# 	def batch_iter():
# 		for i in range(0, len(padded), batch_size):
# 			batch = padded[i: i + batch_size]
# 			labels = [e[0] for e in batch]
# 			processed = [process(e[1]) for e in batch]
# 			# stack into tensor
# 			yield labels, np.array(processed).astype(np.float32)

# 	return batch_iter()

class PickledImageProvider(object):
	def __init__(self, obj_path):
		self.obj_path = obj_path
		self.examples = self.load_pickled_examples()

	def load_pickled_examples(self):
		with open(self.obj_path, "rb") as of:
			examples = list()
			while True:
				try:
					e = pickle.load(of)
					examples.append(e)
					if len(examples) % 10000 == 0:
						print("processed %d examples" % len(examples))
				except EOFError:
					break
				except Exception:
					pass
			print("unpickled total %d examples" % len(examples))

			# byte 파일만 반환하도록
			only_byte_examples = []
			for i in range(3, len(examples)-1, 5):
				only_byte_examples.append(examples[i])

			print("saved total %d examples only for byte"%len(only_byte_examples))
			return only_byte_examples

class FontDataset(torch.utils.data.Dataset):
	def __init__(self, pickled):
		self.path = pickled.obj_path
		self.dset = pickled.examples

	def __getitem__(self, idx):
		img_tuple = self.dset[idx]
		filename, img_byte = img_tuple[0], img_tuple[1]

		filename = filename[:-4]       # 확장자 제거
		filename = filename.split('_') # [카테고리, 폰트번호, 알파벳]

		# 파생변수 생성
		info = {
			'category_vector': np.array([int(i == int(filename[0])) for i in range(5)]),
			'font': int(filename[1]),
			'alphabet_vector': np.array([int(i == int(filename[2])) for i in range(52)])
		}

		# bytes 타입을 numpy array로 변경 후 normalize
		img_arr = np.array(Image.open(io.BytesIO(img_byte)))
		img_arr = normalize_image(img_arr)

		cropped_image, cropped_image_size = tight_crop_image(img_arr, verbose=False)
		centered_image = add_padding(cropped_image, verbose=False)
		
		# 길이를 반환하는 dictionary 추가
		length = {
			'category_vector': len(info['category_vector']),
			'alphabet_vector': len(info['alphabet_vector']),
			'font_vector': len(centered_image)*len(centered_image[0]) # 128x128
		}

		return info, centered_image, length

	def __len__(self):
		return len(self.dset)
	
	
# class NewFontDataset(torch.utils.data.Dataset):
# 	def __init__(self, pickled):
# 		self.path = pickled.obj_path
# 		self.dset = pickled.examples

# 	def __getitem__(self, idx):
# 		info = self.dset[idx]
# 		return info

# 	def __len__(self):
# 		return len(self.dset)

# class TrainDataProvider(object):
# 	"""
# 		train_name = 'train_{카테고리}.obj'
# 		val_name   = 'val_{카테고리}.obj'
# 	"""
# 	def __init__(self, data_dir, train_name, val_name=None, filter_by=None):
# 		self.data_dir = data_dir
# 		self.filter_by = filter_by
# 		self.train_path = os.path.join(self.data_dir, train_name)
# 		self.train = PickledImageProvider(self.train_path)
# 		if val_name:
# 			self.val_path = os.path.join(self.data_dir, val_name)
# 			self.val = PickledImageProvider(self.val_path)
# 		if self.filter_by:
# 			print("filter by label ->", filter_by)
# 			self.train.examples = filter(lambda e: e[0] in self.filter_by, self.train.examples)
# 			if val_name:
# 				self.val.examples = filter(lambda e: e[0] in self.filter_by, self.val.examples)

# 		print("train examples -> %d" % (len(self.train.examples)))
# 		if val_name:
# 			print("val examples -> %d" % (len(self.val.examples)))

# 	def get_train_iter(self, batch_size, shuffle=True):
# 		training_examples = self.train.examples[:]
# 		if shuffle:
# 			np.random.shuffle(training_examples)
# 		return get_batch_iter(training_examples, batch_size, augment=True)

# 	def get_val_iter(self, batch_size, shuffle=True):
# 		"""
# 		Validation iterator runs forever
# 		"""
# 		val_examples = self.val.examples[:]
# 		if shuffle:
# 			np.random.shuffle(val_examples)
# 		while True:
# 			val_batch_iter = get_batch_iter(val_examples, batch_size, augment=False)
# 			for labels, examples in val_batch_iter:
# 				yield labels, examples

# 	def compute_total_batch_num(self, batch_size):
# 		"""Total padded batch num"""
# 		return int(np.ceil(len(self.train.examples) / float(batch_size)))

# 	def get_all_labels(self):
# 		"""Get all training labels"""
# 		return list({e[0] for e in self.train.examples})

# 	def get_train_val_path(self):
# 		return self.train_path, self.val_path

class LatentInfo(torch.utils.data.Dataset):
	def __init__(self, pickled):
		self.path = pickled.obj_path
		self.dset = pickled.examples

	def __getitem__(self, idx):
		img_tuple = self.dset[idx]
		filename, img_byte = img_tuple[0], img_tuple[1]

		filename = filename[:-4]       # 확장자 제거
		filename = filename.split('_') # [폰트 인덱스, 글자 인덱스]

		# 파생변수 생성
		info = {
			'font_index' : int(filename[0]),
			'word_index' : int(filename[1]) 
		}

		# bytes 타입을 numpy array로 변경 후 normalize
		img_arr = np.array(Image.open(io.BytesIO(img_byte)))
		img_arr = normalize_image(img_arr)

		cropped_image, cropped_image_size = tight_crop_image(img_arr, verbose=False)
		centered_image = add_padding(cropped_image, verbose=False)	

		return info, centered_image

	def __len__(self):
		return len(self.dset)


def get_doc2vec():
	vec_10 = pd.read_csv('./dataset/kor/doc2vec_10.csv')
	vec_20 = pd.read_csv('./dataset/kor/doc2vec_20.csv')
	
	# 불필요한 col 제거
	del vec_10['Unnamed: 0']
	del vec_20['Unnamed: 0']
	
	# 폰트 2개 제거 (동화또박, 하나손글씨)
	fonts_ = []
	for font in glob.glob('collection/fonts_kor/*.ttf'):
		fonts_.append(font[21:])
	fonts_ = sorted(fonts_)
	
	vec_10 = vec_10.drop(vec_10.index[25])
	vec_10 = vec_10.drop(vec_10.index[98])
	vec_10 = vec_10.reset_index(drop=True)
	vec_20 = vec_20.drop(vec_20.index[25])
	vec_20 = vec_20.drop(vec_20.index[98])
	vec_20 = vec_20.reset_index(drop=True)
	
	return vec_10, vec_20

	
class KoreanFontDataset(torch.utils.data.Dataset):
	"""
		한글 폰트 클래스. Doc2vec의 vector_size를 명시해주세요(10, 20).
	"""
	def __init__(self, pickled, vector_size=10):
		self.path = pickled.obj_path
		self.dset = pickled.examples
		
		doc2vec = get_doc2vec()
		self.vec = doc2vec[0] if vector_size == 10 else doc2vec[1]
		

	def __getitem__(self, idx):
		img_tuple = self.dset[idx]
		filename, img_byte = img_tuple[0], img_tuple[1]

		filename = filename[:-4]       # 확장자 제거
		filename = filename.split('_') # [폰트 인덱스, 글자 인덱스]
		
		# 파생변수 생성
		font_idx = int(filename[0])
		info = {
			'font_index'  : font_idx,
			'font_doc2vec': self.vec.loc[self.vec.index[font_idx]].tolist(),
			'word_index'  : int(filename[1])
		}
		
		# bytes 타입을 numpy array로 변경 후 normalize
		img_arr = np.array(Image.open(io.BytesIO(img_byte)))
		img_arr = normalize_image(img_arr)

		cropped_image, cropped_image_size = tight_crop_image(img_arr, verbose=False)
		centered_image = add_padding(cropped_image, verbose=False)

		return info, centered_image

	def __len__(self):
		return len(self.dset)