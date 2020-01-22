# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import pickle as pickle
import numpy as np
import pandas as pd
import random
import os
import os.path
import glob
import io

import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.utils import save_image

from PIL import Image
from .utils import pad_seq, bytes_to_file, \
	read_split_image, shift_and_resize_image, normalize_image, \
	tight_crop_image, add_padding

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
    """
		base custom dataset
	"""
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


class LatentInfo(torch.utils.data.Dataset):
    """
		Dataset for embedding latent vector (Conv AE)
	"""
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
			'font_doc2vec': np.array(self.vec.loc[self.vec.index[font_idx]]),
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

class KoreanFontDataset_with_Embedding(torch.utils.data.Dataset):
	"""
	한글 폰트 클래스. Doc2vec의 vector_size를 명시해주세요(10, 20).
	"""
	def __init__(self, pickled, category_emb_path, letter_emb_path): #, vector_size=10):
		self.path = pickled.obj_path
		self.dset = pickled.examples
		self.category_emb_dict = self._load_embedding(category_emb_path)
		self.letter_emb_dict = self._load_embedding(letter_emb_path)

		# doc2vec = get_doc2vec()
		# self.vec = doc2vec[0] if vector_size == 10 else doc2vec[1]


	def __getitem__(self, idx):
		img_tuple = self.dset[idx]
		filename, img_byte = img_tuple[0], img_tuple[1]

		filename = filename[:-4]       # 확장자 제거
		filename = filename.split('_') # [폰트 인덱스, 글자 인덱스]

		# 파생변수 생성
		font_idx = int(filename[0])
		info = {
			'font_index'  : font_idx,
			# 'font_doc2vec': np.array(self.vec.loc[self.vec.index[font_idx]]),
			'word_index'  : int(filename[1])
		}

		# bytes 타입을 numpy array로 변경 후 normalize
		img_arr = np.array(Image.open(io.BytesIO(img_byte)))
		img_arr = normalize_image(img_arr)
		cropped_image, _ = tight_crop_image(img_arr, verbose=False)
		centered_image = add_padding(cropped_image, verbose=False)
		category_vector = self.category_emb_dict[int(filename[0])]
		letter_vector = self.letter_emb_dict[int(filename[1])]

		centered_image = torch.from_numpy(centered_image)
		category_vector = torch.from_numpy(category_vector)
		letter_vector  = torch.from_numpy(letter_vector)

		return info, centered_image, category_vector, letter_vector

	def __len__(self):
		return len(self.dset)

	def _load_embedding(self, pkl_path):
		with open(pkl_path, 'rb') as f:
			emb_dict = pickle.load(f)
		return emb_dict

class CategoryDataset(torch.utils.data.Dataset):
	def __init__(self, pickled):
		self.path = pickled.obj_path
		self.dset = pickled.examples

	def __getitem__(self, idx):
		# 8개 글자를 불러온다.
		imgs = list()
		for i in range(0+idx, len(self.dset), 107): # 8
			imgs.append(self.dset[i])

		imgs_byte = [img[1] for img in imgs]
		imgs_arr = [ np.array(Image.open(io.BytesIO(img_byte))) for img_byte in imgs_byte ]
		imgs_arr = [ normalize_image(img_arr) for img_arr in imgs_arr ]

		cropped_images = [ tight_crop_image(img_arr, verbose=False)[0] for img_arr in imgs_arr ]
		centered_images = [ add_padding(cropped_image, verbose=False) for cropped_image in cropped_images ]

		return np.array(centered_images)

	def __len__(self):
		return len(self.dset/8)

def default_image_loader(path):
	return Image.open(path).convert('RGB')

class TripletImageLoader(torch.utils.data.Dataset): 
	def __init__(self, pickled, triplets_file_name, base_path=None, filenames_filename=None, transform=None,
				 loader=default_image_loader):
		""" 
		filenames_filename: 
			A text file with each line containing the path to an image e.g.,
			images/class1/sample.jpg

		triplets_file_name: 
			A text file with each line containing three integers, 
			where integer i refers to the i-th image in the filenames file. 
			For a line of intergers 'a b c', a triplet is defined such that image a is more 
			similar to image c than it is to image b, 
			e.g., 0 2017 42 
		"""
		self.dset = pickled.examples
		triplets = []
		anchor_labels = [] #
		for line in open(triplets_file_name):
			triplets.append((line.split()[0], line.split()[1], line.split()[2])) # anchor, far, close
			anchor_labels.append(int(line.split()[3])) #
		self.triplets = triplets
		self.labels = anchor_labels #
		self.transform = transform
		self.loader = loader

	def __getitem__(self, index):
		path1, path2, path3 = self.triplets[index]
		anchor_label = self.labels[index]
		img1_tuple = self.dset[int(path1)]
		img2_tuple = self.dset[int(path2)]
		img3_tuple = self.dset[int(path3)]

		info = {                         # clustering을 위해 anchor_index도 추가하였다.
			'anchor_index': int(path1),
			'anchor_label': anchor_label
		}

		# byte만 사용할 예정
		img1, byte_1 = img1_tuple[0], img1_tuple[1]
		img2, byte_2 = img2_tuple[0], img2_tuple[1]
		img3, byte_3 = img3_tuple[0], img3_tuple[1]

		# bytes 타입을 numpy array로 변경 후 normalize
		img_arr_1 = np.array(Image.open(io.BytesIO(byte_1)))
		img_arr_1 = normalize_image(img_arr_1)

		img_arr_2 = np.array(Image.open(io.BytesIO(byte_2)))
		img_arr_2 = normalize_image(img_arr_2)

		img_arr_3 = np.array(Image.open(io.BytesIO(byte_3)))
		img_arr_3 = normalize_image(img_arr_3)

		cropped_image_1, cropped_image_size_1 = tight_crop_image(img_arr_1, verbose=False)
		centered_image_1 = add_padding(cropped_image_1, verbose=False)

		cropped_image_2, cropped_image_size_2 = tight_crop_image(img_arr_2, verbose=False)
		centered_image_2 = add_padding(cropped_image_2, verbose=False)

		cropped_image_3, cropped_image_size_3 = tight_crop_image(img_arr_3, verbose=False)
		centered_image_3 = add_padding(cropped_image_3, verbose=False)

		return (centered_image_1, centered_image_2, centered_image_3), info #

	def __len__(self):
		return len(self.triplets)