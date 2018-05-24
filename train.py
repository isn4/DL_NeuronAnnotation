# Original Code by Marko Jocić found here: https://github.com/jocicmarko/ultrasound-nerve-segmentation

# All changes and updates made by Kristi Bushman and Ishtar Nyawira

# NOTE: 35,000 images take roughly 11 hours to complete training

from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
import h5py
import tensorflow as tf

from data import load_train_data, load_test_data

# change this to the directory where you want to keep your predicted images
prediction_path = '/your/path/here/for/predicted/images/' # don't forget final '/'

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 352
img_cols = 480

smooth = 1.

weight_path = 'weights/weights2.h5'

def mean_iu(y_true, y_pred):
	y_true_f = K.flatten(y_true) 
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	union = K.sum(y_true_f) + K.sum(y_pred_f)
	return (intersection + smooth) / (union - intersection + smooth)

def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true) 
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	union = K.sum(y_true_f) + K.sum(y_pred_f)
	return (2. * intersection + smooth) / (union + smooth)


def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)


def get_unet():
	inputs = Input((img_rows, img_cols, 3))
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

	conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

	# formerly metrics=dice_coef
	# learning rate changed from 1e-5
	model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[mean_iu])

	return model


# def preprocess_images(imgs):
#     imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols,3), dtype=np.uint8)
#     for i in range(imgs.shape[0]):
#         imgs_p[i] = resize(imgs[i], (img_cols, img_rows,3), preserve_range=True)

#     imgs_p = imgs_p[..., np.newaxis]
#     return imgs_p

# def preprocess_masks(imgs):
#     imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
#     for i in range(imgs.shape[0]):
#         imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

#     imgs_p = imgs_p[..., np.newaxis]
#     return imgs_p


def train_and_predict():

	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)
	imgs_train, imgs_mask_train = load_train_data()
	imgs_mask_train = np.bool_(imgs_mask_train)
	imgs_mask_train = ~imgs_mask_train # switch 1 and 0 so in format that works correctly with dice coefficient

	# imgs_train = preprocess_images(imgs_train)
	# imgs_mask_train = preprocess_masks(imgs_mask_train)

	# imgs_train = imgs_train.astype('float32')
	# mean = np.mean(imgs_train, axis=3)  # mean for data centering
	# std = np.std(imgs_train, axis=3)  # std for data normalization

	# imgs_train -= mean
	# imgs_train /= std

	# imgs_mask_train = imgs_mask_train.astype('float32')
	# imgs_mask_train /= 255.  # scale masks to [0, 1]

	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	model = get_unet()
	model_checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', save_best_only=True)
	tensorboard = TensorBoard(log_dir='./logs/3', histogram_freq=1, write_graph=True)

	print('-'*30)
	print('Fitting model...')
	print('-'*30)
	model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
			  validation_split=0, # <- changed from 0.2 because of ResourceExhaustedError during val
			  callbacks=[model_checkpoint, tensorboard])

	print('-'*30)
	print('Loading and preprocessing test data...')
	print('-'*30)
	imgs_test, imgs_id_test = load_test_data()
	# imgs_test = preprocess_imgs(imgs_test)

	# imgs_test = imgs_test.astype('float32')
	# imgs_test -= mean
	# imgs_test /= std

	print('-'*30)
	print('Loading saved weights...')
	print('-'*30)
	model.load_weights(weight_path)

	print('-'*30)
	print('Predicting masks on test data...')
	print('-'*30)
	imgs_mask_test = model.predict(imgs_test, verbose=1)
	np.save('rg35K_test_mask_preds3.npy', imgs_mask_test)  # name the name according to annotation type

	print('-' * 30)
	print('Saving predicted masks to files...')
	print('-' * 30)
	pred_dir = prediction_path + 'preds_rg/'
	if not os.path.exists(pred_dir):
		os.mkdir(pred_dir)
	for image, image_id in zip(imgs_mask_test, imgs_id_test):
		image = (image[:, :, 0] * 255.).astype(np.uint8)
		savepath = os.path.join(pred_dir, str(image_id) + '_pred.png')
		savepath = savepath.replace('gs/', '')
		imsave(savepath, image)

# if __name__ == '__main__':
#     train_and_predict()

train_and_predict()

