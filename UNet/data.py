# Original Code by Marko Jocić found here: https://github.com/jocicmarko/ultrasound-nerve-segmentation

# All changes and updates made by Kristi Bushman and Ishtar Nyawira

from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread
from PIL import Image

# create a list of full absolute paths in text documents for both the training and testing set
# the lists should have original and mask images space-separated and different images, newline separated as such:
	# /full/path/original_0.png /full/path/mask_0.png
	# /full/path/original_1.png /full/path/mask_1.png
	# /full/path/original_2.png /full/path/mask_2.png

# change this to the paths where you're keeping your training & testing set text files
train_txt_paths = '/your/path/here/for/training/images/rg_train_35K.txt'
test_txt_paths = '/your/path/here/for/test/images/test.txt'

image_rows = 360
image_cols = 480
rg_total = 35000
flood_total = 35000
manual_total = 3838
test_total = 50


def create_train_data(total):

	imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
	imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

	i = 0
	print('-'*30)
	print('Creating training images...')
	print('-'*30)
	with open(train_txt_paths) as fp:
		for line in fp:
			img_path, img_mask_path = line.split(" ")
			img_mask_path = img_mask_path.split("\n")[0]
			img = Image.open(img_path)
			img = img.convert('RGB')
			img_mask = imread(img_mask_path, as_grey=True)

			imgs[i] = img
			imgs_mask[i] = img_mask

			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, total))
			i += 1
	print('Loading done.')

	np.save('rg_imgs_train.npy', imgs)   # <-- change to a name that fits the current annotation type
	np.save('rg_imgs_mask_train.npy', imgs_mask)   # <-- change to a name that fits the current annotation type
	print('Saving to .npy files done.')


def load_train_data():
	# change these to a name that fits the current annotation type
	imgs_train = np.load('rg35K_imgs_train.npy')[:,:352,:,:]
	imgs_mask_train = np.expand_dims(np.load('rg35K_imgs_mask_train.npy')[:,:352,:], axis=3)
	return imgs_train, imgs_mask_train


def create_test_data(total):
	imgs = np.ndarray((total, image_rows, image_cols,3), dtype=np.uint8)
	imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
	imgs_id = np.ndarray((total, ), dtype='object')
 
	i = 0
	print('-'*30)
	print('Creating test images...')
	print('-'*30)
	with open(test_txt_paths) as fp:
		for line in fp:
			img_path, img_mask_path = line.split(" ")
			img_mask_path = img_mask_path.split("\n")[0]
			img = Image.open(img_path)
			img = img.convert('RGB')
			img_mask = imread(img_mask_path, as_grey=True)
			img_id = img_path[36:].split('.')[0]

			imgs[i] = img
			imgs_id[i] = img_id
			imgs_mask[i] = img_mask

			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, total))
			i += 1
	print('Loading done.')

	# change to a name that fits your needs
	np.save('50_imgs_test.npy', imgs)
	np.save('50_imgs_id_test.npy', imgs_id)
	np.save('50_imgs_mask_test.npy', imgs_mask)
	print('Saving to .npy files done.')


def load_test_data():
	# change to a name that fits your needs
	imgs_test = np.load('50_imgs_test.npy')[:,:352,:,:]
	imgs_id = np.load('50_imgs_id_test.npy')
	return imgs_test, imgs_id

if __name__ == '__main__':
	create_train_data(flood_total)
	create_test_data(test_total)
