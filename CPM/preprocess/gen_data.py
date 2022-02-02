# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import glob
import os
import torch
import torch.utils.data
import torchvision.transforms.functional
import cv2


def read_dataset(path):
	"""
	Read training dataset or validation dataset.
	:param path: The path of dataset.
	:return: The list of filenames.
	"""

	"""
	path : 'F:/Python/PyCharmWorkspace/CPM/lspet/' 
	os.path.join(path, 'images/*.jpg') 결과는 >>> F:/Python/PyCharmWorkspace/CPM/lspet/images/*.jpg
	"""
	image_list = glob.glob(os.path.join(path, 'images/*.jpg')) #path와 images/*.jpg 를 결합하여 1개의 경로로 만든다.
	return image_list # images 파일 안 .jpg확장자를 가진 파일들을 리스트 형태로 모든 변수 : image_list


def read_mat(mode, path, image_list):
	"""
	Read joints.mat file.
	joints.mat in lspet is (14, 3, 10000); joints.mat in lsp is (3, 14, 2000)
	:param mode: 'lspet(training dataset)' or 'lsp(validation dataset)'
	:param path: The path of joints.mat.
	:param image_list: The array of image filenames.
	:return:
	"""

	"""
	path : 'F:/Python/PyCharmWorkspace/CPM/lspet/' 
	image_list 는 read_dataset 함수에서의 image_list
	"""
	mat_arr = sio.loadmat(os.path.join(path, 'joints.mat'))['joints'] # mat (matlab) 확장자 파일 불러오기
	# (x,y,z)
	# LSPET: z = 1 means the key points is not blocked.
	# LSP: z = 0 means the key points is not blocked.
	key_point_list = []
	limits = []

	if mode == 'lspet': # training dataset
		key_point_list = np.transpose(mat_arr, (2, 0, 1)).tolist() # axis2, axis0, axis1 형태로 행렬이 변경 => (10000, 14, 3)

		# Calculate the limits to find center points
		limits = np.transpose(mat_arr, (2, 1, 0)) # axis2, axis1, axis0 형태로 행렬이 변경 => # (10000, 3, 14)

	if mode == 'lsp': # validation dataset
		# Guarantee z = 1 means the key points is not blocked
		mat_arr[2] = np.logical_not(mat_arr[2]) #mat_arr[2] = z, z가 아닌 것들
		key_point_list = np.transpose(mat_arr, (2, 1, 0)).tolist() # axis2, axis0, axis1 형태로 행렬이 변경 => (2000, 14, 3)
		# Calculate the limits to find center points
		limits = np.transpose(mat_arr, (2, 0, 1))  # axis2, axis0, axis1 형태로 행렬이 변경 => (2000, 3, 14)

	center_point_list = []
	scale_list = []

	for i in range(limits.shape[0]): #limits.shape[0] =  10000장/2000장 이미지 개수
		image = cv2.imread(image_list[i]) #i번째 이미지 읽기

		h = image.shape[0] #높이
		w = image.shape[1] #width 가로

		# Calculate the center points of each image #center points 구하기
		center_x = (limits[i][0][limits[i][0] > 0].min() + limits[i][0][limits[i][0] < w].max()) / 2
		"""
		limits[i][0] : limits[i] 10000장 중 i번째의 [0] 즉, i번째의 (x,y,z) 중 x 
		limits[i][0][limits[i][0] > 0].min() : i번째의 (x,y,z) x 중 0보다 큰 값들의 min() / i번째의 x값들중 양수인 값들 중에서의 가장 작은 값
		
		limits[i][0][limits[i][0] < w].max() : i번째의 x값들중 w보다 작은 값들 중에서의 가장 큰 값
		
		이 둘의 합을 /2 한것이 center point의 x값
		"""
		center_y = (limits[i][1][limits[i][1] > 0].min() + limits[i][1][limits[i][1] < h].max()) / 2
		"""
		limits[i][1] : limits[i] 10000장 중 i번째의 [1] 즉, i번째의 (x,y,z) 중 y
		limits[i][1][limits[i][1] > 0].min() : i번째의 (x,y,z) y 중 0보다 큰 값들의 min() / i번째의 y값들 중 양수인 값들 중에서의 가장 작은 값

		limits[i][1][limits[i][1] < h].max() : i번째의 y값들 중 h보다 작은 값들 중에서의 가장 큰 값

		이 둘의 합을 /2 한것이 center point의 y값
		"""
		center_point_list.append([center_x, center_y]) # 각 img들의 center points들을 center_point_list에 담기

		# Calculate the scale of each image
		scale = (limits[i][1][limits[i][1] < h].max() - limits[i][1][limits[i][1] > 0].min() + 4) / 368
		"""
		limits[i][1][limits[i][1] < h].max() i번째 사진의 y값 중 h보다 작은 것들 중 max 값 
		limits[i][1][limits[i][1] > 0].min() i번째 사진의 y값 중 0보다 큰 것들 중 min 값
		
		큰 값에서 작은 값을 뺀 후 +4를 하고 /368 로 나누어 준다. 												# scaling 할때 + 4 는 갑자기 왜하는건지..?
		"""
		scale_list.append(scale) # scale들을 담는 scale_list

	return key_point_list, center_point_list, scale_list # center_point_list에 [center_x, center_y] 가 들어있다.


def gaussian_kernel(size_w, size_h, center_x, center_y, sigma):
	grid_y, grid_x = np.mgrid[0:size_h, 0:size_w]
	D2 = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2 # gird x와 center_x 의 차이 제곱과 grid y와 center_y의 차이 제곱의 합

	return np.exp(-D2 / 2.0 / sigma / sigma)


'''
from train.py
training_dataset_path = 'F:/Python/PyCharmWorkspace/CPM/lspet/'
data = LSP_DATA('lspet', training_dataset_path, 8, Compose([RandomResized(), RandomCrop(368)]))
'''
class LSP_DATA(torch.utils.data.Dataset):
	def __init__(self, mode, path, stride, transformer=None):
		self.image_list = read_dataset(path)
		self.key_point_list, self.center_point_list, self.scale_list = read_mat(mode, path, self.image_list)
		self.stride = stride
		self.transformer = transformer
		self.sigma = 3.0 # gaussian kernel에 쓰일 sigma값

	def __getitem__(self, item):
		image_path = self.image_list[item]
		image = np.array(cv2.imread(image_path), dtype=np.float32)

		# 해당 item의 key_point, center_points, scale을 저장
		key_points = self.key_point_list[item]
		center_points = self.center_point_list[item]
		scale = self.scale_list[item]

		# Expand dataset
		image, key_points, center_points = self.transformer(image, key_points, center_points, scale)
		h, w, _ = image.shape

		# Generate heatmap
		size_h = int(h / self.stride) # heatmap의 h 사이즈
		size_w = int(w / self.stride) # heatmap의 w 사이즈
		heatmap = np.zeros((size_h, size_w, len(key_points) + 1), dtype=np.float32) # len(key_points) + 1 : 관절 개수 + back ground

		# Generate the heatmap of all key points / 모든 key point에 대해 heatmap 만들기 (key point = 관절 개수)
		for i in range(len(key_points)):																			# len(key_points) 는 2000/10000?
			# Resize image from 368 to 46
			x = int(key_points[i][0]) * 1.0 / self.stride
			y = int(key_points[i][1]) * 1.0 / self.stride

			kernel = gaussian_kernel(size_h=size_h, size_w=size_w, center_x=x, center_y=y, sigma=self.sigma)
			kernel[kernel > 1] = 1 #kernel값이 1보다 큰 곳에는 1
			kernel[kernel < 0.01] = 0 #kernel값이 0.01보다 작은 곳에는 0
			heatmap[:, :, i + 1] = kernel # 1과 0의 값 그리고 1, 0.01사이 값은 소수점으로 구성 / 배경이 아닌 것에 대해 값 넣기

		# Generate the heatmap of background (back ground의 heatmap) 0번째
		heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)

		# Generate center map (center of human body picture)
		centermap = np.zeros((h, w, 1), dtype=np.float32)
		kernel = gaussian_kernel(size_h=h, size_w=w, center_x=center_points[0], center_y=center_points[1], sigma=self.sigma)
		kernel[kernel > 1] = 1
		kernel[kernel < 0.01] = 0
		centermap[:, :, 0] = kernel

		image -= image.mean()

		image = torchvision.transforms.functional.to_tensor(image)
		heatmap = torch.from_numpy(np.transpose(heatmap, (2, 0, 1)))
		centermap = torch.from_numpy(np.transpose(centermap, (2, 0, 1)))

		return image.float(), heatmap.float(), centermap.float()

	def __len__(self):
		return len(self.image_list)
