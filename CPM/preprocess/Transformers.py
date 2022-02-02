import cv2
import random
import numbers
import numpy as np


def resize_image_points(image, key_points, center_points, ratio):
	"""
	Resize image, key points and center points according the ratio
	:param image: The image to be resized.
	:param key_points: The key points to be resized.
	:param center_points: The center points to be resized.
	:param ratio: The ratio used to resize image and points.
	:return: Resized image, key points and center points
	"""
	if isinstance(ratio, numbers.Number):
		# Train
		num = len(key_points)
		for i in range(num): # 10000개의 img
			key_points[i][0] *= ratio # ratio만큼 i번째 그림의 key_point의 x좌표에 곱해주기
			key_points[i][1] *= ratio # ratio만큼 i번째 그림의 key_point의 y좌표에 곱해주기
		center_points[0] *= ratio # ratio만큼 i번째 그림의 center_point의 x좌표에 곱해주기
		center_points[1] *= ratio # ratio만큼 i번째 그림의 center_point의 y좌표에 곱해주기

		return cv2.resize(image, (0, 0), fx=ratio, fy=ratio), key_points, center_points
	else:
		# Validation
		num = len(key_points)
		for i in range(num):
			key_points[i][0] *= ratio[0]
			key_points[i][1] *= ratio[1]
		center_points[0] *= ratio[0]
		center_points[1] *= ratio[1]

		return cv2.resize(image, (368, 368), interpolation=cv2.INTER_CUBIC), key_points, center_points


class RandomResized(object):
	"""Randomly resized given numpy array"""

	def __init__(self, scale_min=0.3, scale_max=1.1):
		self.scale_min = scale_min
		self.scale_max = scale_max

	def __call__(self, image, key_points, center_points, scale):
		"""
		Randomly resized given numpy array
		:param image: The image to be resized.
		:param key_points: The key points to be resized.
		:param center_points: The center points to be resized.
		:param scale: The scale of each image, representing the main part, which is calculated at line 63 of gen_data.py.
		:return: Randomly resize image, key points and center points.
		"""
		random_scale = random.uniform(self.scale_min, self.scale_max)
		ratio = random_scale / scale #random의 ratio 계산

		return resize_image_points(image, key_points, center_points, ratio)


class TestResized(object):
	def __init__(self, size):
		self.size = (size, size)

	def __call__(self, image, key_points, center_points):
		height, width, _ = image.shape
		ratio = ((self.size[0] * 1.0) / width, (self.size[1] * 1.0) / height)

		return resize_image_points(image, key_points, center_points, ratio)


def crop(image, key_points, center_points, offset_left, offset_up, new_height, new_width):
	num = len(key_points)
	for i in range(num): # key_points[i][0] = x, key_points[i][1] = y, key_points[i][2] = z
		if key_points[i][2] == 0:
			# The key points is blocked
			continue
		# Translate key points
		key_points[i][0] -= offset_left # offset_left 만큼 x 길이 빼주기
		key_points[i][1] -= offset_up # offset_up 만큼 y 길이 빼주기

	# Translate center points
	center_points[0] -= offset_left
	center_points[1] -= offset_up

	# Get the width and height of original image
	ori_height, ori_width, _ = image.shape

	new_image = np.empty((new_width, new_height, 3), dtype=np.float32)
	new_image.fill(128) # 128로 채우기?

	# The coordinates of new image
	start_x = 0
	end_x = new_width
	start_y = 0
	end_y = new_height

	# The coordinates of original image  																	#이부분 잘 이해 안됨...
	ori_start_x = offset_left
	ori_end_x = ori_start_x + new_width
	ori_start_y = offset_up
	ori_end_y = ori_start_y + new_height

	if offset_left < 0:
		# start_x should be added, because center points and key points are added
		start_x = -offset_left
		ori_start_x = 0
	if ori_end_x > ori_width:
		end_x = ori_width - offset_left
		ori_end_x = ori_width

	if offset_up < 0:
		start_y = -offset_up
		ori_start_y = 0
	if ori_end_y > ori_height:
		end_y = ori_height - offset_up
		ori_end_y = ori_height

	new_image[start_y: end_y, start_x: end_x, :] = image[ori_start_y: ori_end_y, ori_start_x: ori_end_x, :].copy()

	return new_image, key_points, center_points


class RandomCrop(object):
	def __init__(self, size):
		self.size = (size, size)  # (368, 368)

	def __call__(self, image, key_points, center_points):
		x_offset = random.randint(-5, 5)
		y_offset = random.randint(-5, 5)
		center_x = center_points[0] + x_offset		# center_points random하게 crop하기
		center_y = center_points[1] + y_offset

		offset_left = int(round(center_x - self.size[0] / 2))
		offset_up = int(round(center_y - self.size[1] / 2))

		return crop(image, key_points, center_points, offset_left, offset_up, self.size[0], self.size[1])


class Compose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, img, kpt, center, scale=None):
		for t in self.transforms:
			if isinstance(t, RandomResized):
				img, kpt, center = t(img, kpt, center, scale)
			else:
				img, kpt, center = t(img, kpt, center)

		return img, kpt, center
