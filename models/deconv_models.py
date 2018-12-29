#coding: utf-8
from __future__ import print_function
import sys

sys.path += ['../layers/']

import numpy as np
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Input
from keras.layers.convolutional import ZeroPadding2D
from pool_unpool import UndoMaxPooling2D
from deconv2D import Deconv2D
import torch

## TODO Models are not yet checked and tested! (check the filter sizes for deconv filters)

sz = 32
num_classes = 1000

#If you want to reconstruct from a single feature map / activation, you can
# simply set all the others to 0.

# The Deconv2D layers should have the same name as the associated Conv2D layers.
# The shapes can be extracted from [model to deconvolve].summary().

## CREDIT: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
def VGG_16(pretrained=True, weights_path=None, noutputs=num_classes):
	if (pretrained):
		weights_path = './data/weights/vgg16_weights.h5'

	inp = Input(batch_shape = (1, sz // 4, sz // 4, 128*2*2))
	x = inp

	##TODO
	pos2 = Input(batch_shape = (1, sz // 4, sz // 4, 32))
	x = UndoMaxPooling2D((1, sz, sz, 32), name="pool2")([x, pos2])

	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(512, (3, 3), activation='relu', name="conv5-1")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(512, (3, 3), activation='relu', name="conv5-2")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(512, (3, 3), activation='relu', name="conv5-3")(x)
	x, pos5 = MaxPooling2D(pool_size=2, strides=2, name="pool5")(x)

	pos2 = Input(batch_shape = (1, sz // 4, sz // 4, 32))
	x = UndoMaxPooling2D((1, sz, sz, 32), name="pool2")([x, pos2])
	x = Deconv2D(64//2,3,padding='SAME',activation='relu', name="conv2-2")(x)
	x = Deconv2D(64//2,3,padding='SAME',activation='relu', name="conv2-1")(x)

	pos1 = Input(batch_shape = (1, sz // 2, sz // 2, 16))
	x = UndoMaxPooling2D((1, sz, sz, 16), name="pool1")([x, pos1])
	x = Deconv2D(32//2, 3, padding='SAME', activation="relu", name="conv1-2")(x)
	x = Deconv2D(32//2, 3, padding='SAME', activation="relu", name="conv1-1")(x)

	model = Model(inputs = [inp, pos1, pos2, pos3, pos4, pos5], outputs = x)



	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(64, (3, 3), activation='relu', name="conv1-1")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(64, (3, 3), activation='relu', name="conv1-2")(x)
	x, pos1 = MaxPooling2D(pool_size=2, strides=2, name="pool1")(x)

	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(128, (3, 3), activation='relu', name="conv2-1")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(128, (3, 3), activation='relu', name="conv2-2")(x)
	x, pos2 = MaxPooling2D(pool_size=2, strides=2, name="pool2")(x)

	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(256, (3, 3), activation='relu', name="conv3-1")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(256, (3, 3), activation='relu', name="conv3-2")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(256, (3, 3), activation='relu', name="conv3-3")(x)
	x, pos3 = MaxPooling2D(pool_size=2, strides=2, name="pool3")(x)
	
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(256, (3, 3), activation='relu', name="conv4-1")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(256, (3, 3), activation='relu', name="conv4-2")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(256, (3, 3), activation='relu', name="conv4-3")(x)
	x, pos4 = MaxPooling2D(pool_size=2, strides=2, name="pool4")(x)





	if weights_path:
		model.load_weights(weights_path, by_name = True)

	return model

## CREDIT: https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/
def Conv2(pretrained=True, weights_path=None, noutputs=num_classes):
	if (pretrained):
		weights_path = './data/weights/conv2_weights.h5'

	inp = Input(batch_shape = (1, sz // 4, sz // 4, 64))
	x = inp

	pos2 = Input(batch_shape = (1, sz // 4, sz // 4, 32))
	x = UndoMaxPooling2D((1, sz, sz, 32), name="pool2")([x, pos2])
	x = Deconv2D(64//2,3,padding='SAME',activation='relu', name="conv2-2")(x)
	x = Deconv2D(64//2,3,padding='SAME',activation='relu', name="conv2-1")(x)

	pos1 = Input(batch_shape = (1, sz // 2, sz // 2, 16))
	x = UndoMaxPooling2D((1, sz, sz, 16), name="pool1")([x, pos1])
	x = Deconv2D(32//2, 3, padding='SAME', activation="relu", name="conv1-2")(x)
	x = Deconv2D(32//2, 3, padding='SAME', activation="relu", name="conv1-1")(x)

	model = Model(inputs = [inp, pos1, pos2], outputs = x)

	if weights_path:
		model.load_weights(weights_path, by_name=True)

	return model

## CREDIT: https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/
def Conv(pretrained=True, weights_path=None, noutputs=num_classes):
	if (pretrained):
		weights_path = './data/weights/conv_weights.h5'

	inp = Input(batch_shape = (1, sz // 4, sz // 4, 128))
	x = inp

	pos3 = Input(batch_shape = (1, sz // 4, sz // 4, 64))
	x = UndoMaxPooling2D((1, sz, sz, 64), name="pool6")([x, pos3])
	x = Deconv2D(128//2,3,padding='SAME',activation='relu', name="conv6")(x)
	x = Deconv2D(128//2,3,padding='SAME',activation='relu', name="conv5")(x)

	pos2 = Input(batch_shape = (1, sz // 2, sz // 2, 32))
	x = UndoMaxPooling2D((1, sz, sz, 32), name="pool4")([x, pos2])
	x = Deconv2D(64//2, 3, padding='SAME', activation="relu", name="conv4")(x)
	x = Deconv2D(64//2, 3, padding='SAME', activation="relu", name="conv3")(x)

	pos1 = Input(batch_shape = (1, sz // 2, sz // 2, 16))
	x, pos1 = UndoMaxPooling2D((1, sz, sz, 16), name="pool2")([x, pos1])
	x = Deconv2D(32//2, 3, padding='SAME', activation="relu", name="conv2")(x)
	x = Deconv2D(32//2, 3, padding='SAME', activation='relu', name="conv1")(x)

	model = Model(inputs = [inp, pos1, pos2, pos3], outputs = x)

	if weights_path:
		model.load_weights(weights_path, by_name=True)

	return model

