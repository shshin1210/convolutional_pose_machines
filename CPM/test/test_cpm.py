import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F
from preprocess.gen_data import gaussian_kernel


def get_key_points(heatmap6, height, width):
	"""
	Get all key points from heatmap6.
	:param heatmap6: The heatmap6 of CPM cpm.
	:param height: The height of original image.
	:param width: The width of original image.
	:return: All key points of the person in the original image.
	"""
	# Get final heatmap
	heatmap = np.asarray(heatmap6.cpu().data)[0]

	key_points = []
	# Get k key points from heatmap6
	for i in heatmap[1:]:
		# Get the coordinate of key point in the heatmap (46, 46)
		y, x = np.unravel_index(np.argmax(i), i.shape)

		# Calculate the scale to fit original image
		scale_x = width / i.shape[1]
		scale_y = height / i.shape[0]
		x = int(x * scale_x)
		y = int(y * scale_y)

		key_points.append([x, y])

	return key_points


def draw_image(image, key_points):
	"""
	Draw limbs in the image.
	:param image: The test image.
	:param key_points: The key points of the person in the test image.
	:return: The painted image.
	"""
	'''ALl limbs of person:
	head top->neck
	neck->left shoulder
	left shoulder->left elbow
	left elbow->left wrist
	neck->right shoulder
	right shoulder->right elbow
	right elbow->right wrist
	neck->left hip
	left hip->left knee
	left knee->left ankle
	neck->right hip
	right hip->right knee
	right knee->right ankle
	'''
	limbs = [[13, 12], [12, 9], [9, 10], [10, 11], [12, 8], [8, 7], [7, 6], [12, 3], [3, 4], [4, 5], [12, 2], [2, 1],
	         [1, 0]]

	# draw key points
	for key_point in key_points:
		x = key_point[0] #key_point x좌표
		y = key_point[1] #key_point y좌표
		cv2.circle(image, (x, y), radius=1, thickness=-1, color=(0, 0, 255))

	# draw limbs
	for limb in limbs:
		start = key_points[limb[0]] #keypoints[limb의 x좌표] #예를 들어 keypoints[13]과 keypoints[12]가 연결
		end = key_points[limb[1]] #keypoints[limb의 y좌표]
		color = (0, 0, 255)  # BGR
		cv2.line(image, tuple(start), tuple(end), color, thickness=1, lineType=4)

	return image


if __name__ == "__main__":
	model = torch.load('../model/best_cpm.pth').cuda()

	image_path = '../test_data/test4.jpg'
	image = cv2.imread(image_path)
	height, width, _ = image.shape
	image = np.asarray(image, dtype=np.float32)
	image = cv2.resize(image, (368, 368), interpolation=cv2.INTER_CUBIC)

	# Normalize
	image -= image.mean()
	image = F.to_tensor(image)

	# Generate center map
	centermap = np.zeros((368, 368, 1), dtype=np.float32)
	kernel = gaussian_kernel(size_h=368, size_w=368, center_x=184, center_y=184, sigma=3)
	kernel[kernel > 1] = 1
	kernel[kernel < 0.01] = 0
	centermap[:, :, 0] = kernel
	centermap = torch.from_numpy(np.transpose(centermap, (2, 0, 1)))

	image = torch.unsqueeze(image, 0).cuda()
	centermap = torch.unsqueeze(centermap, 0).cuda()

	model.eval()
	input_var = torch.autograd.Variable(image)
	center_var = torch.autograd.Variable(centermap)

	heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var, center_var)
	key_points = get_key_points(heat6, height=height, width=width)

	image = draw_image(cv2.imread(image_path), key_points)

	cv2.imshow('test image', image)
	cv2.waitKey(0)

	cv2.imwrite(image_path.rsplit('.', 1)[0] + '_ans.jpg', image)
