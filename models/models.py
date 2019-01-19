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
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from pool_unpool import MaxPooling2D
from keras.backend import tf as ktf
import torch

num_classes = 1000
msg = "* Loaded weights! (CNN)"

from keras.applications.vgg16 import VGG16

## Get weights for VGG16
#model = VGG16(include_top=True, weights='imagenet', classes=num_classes)
#model.save_weights('vgg16_weights.h5')

## CREDIT: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
def VGG_16(pretrained=True, weights_path=None, noutputs=num_classes, deconv=False, sz=224, layer=""):
	if (pretrained and not weights_path):
		weights_path = './data/weights/vgg16_weights.h5'

	inp = Input(shape = (sz, sz, 3), name="input")
	x = inp
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(64, (3, 3), name="block1_conv1")(x)
	layers = ["block1_conv1"]
	if (not layer in layers):
		x = Activation('relu')(x)
		x = ZeroPadding2D((1, 1))(x)
		x = Conv2D(64, (3, 3), name="block1_conv2")(x)
	layers.append("block1_conv2")
	if (not layer in layers):
		x = Activation('relu')(x)
		x, pos1 = MaxPooling2D(pool_size=2, strides=2, name="block1_pool")(x)
	layers.append("block1_pool")
	if (not layer in layers):
		x = ZeroPadding2D((1, 1))(x)
		x = Conv2D(128, (3, 3), name="block2_conv1")(x)
	layers.append("block2_conv1")
	if (not layer in layers):
		x = Activation('relu')(x)
		x = ZeroPadding2D((1, 1))(x)
		x = Conv2D(128, (3, 3), name="block2_conv2")(x)
	layers.append("block2_conv2")
	if (not layer in layers):
		x = Activation('relu')(x)
		x, pos2 = MaxPooling2D(pool_size=2, strides=2, name="block2_pool")(x)
	layers.append("block2_pool")
	if (not layer in layers):
		x = ZeroPadding2D((1, 1))(x)
		x = Conv2D(256, (3, 3), name="block3_conv1")(x)
	layers.append("block3_conv1")
	if (not layer in layers):
		x = Activation('relu')(x)
		x = ZeroPadding2D((1, 1))(x)
		x = Conv2D(256, (3, 3), name="block3_conv2")(x)
	layers.append("block3_conv2")
	if (not layer in layers):
		x = Activation('relu')(x)
		x = ZeroPadding2D((1, 1))(x)
		x = Conv2D(256, (3, 3), name="block3_conv3")(x)
	layers.append("block3_conv3")
	if (not layer in layers):
		x = Activation('relu')(x)
		x, pos3 = MaxPooling2D(pool_size=2, strides=2, name="block3_pool")(x)
	layers.append("block3_pool")
	if (not layer in layers):	
		x = ZeroPadding2D((1, 1))(x)
		x = Conv2D(512, (3, 3), name="block4_conv1")(x)
	layers.append("block4_conv1")
	if (not layer in layers):
		x = Activation('relu')(x)
		x = ZeroPadding2D((1, 1))(x)
		x = Conv2D(512, (3, 3), name="block4_conv2")(x)
	layers.append("block4_conv2")
	if (not layer in layers):
		x = Activation('relu')(x)
		x = ZeroPadding2D((1, 1))(x)
		x = Conv2D(512, (3, 3), name="block4_conv3")(x)
	layers.append("block4_conv3")
	if (not layer in layers):
		x = Activation('relu')(x)
		x, pos4 = MaxPooling2D(pool_size=2, strides=2, name="block4_pool")(x)
	layers.append("block4_pool")
	if (not layer in layers):
		x = ZeroPadding2D((1, 1))(x)
		x = Conv2D(512, (3, 3), name="block5_conv1")(x)
	layers.append("block5_conv1")
	if (not layer in layers):
		x = Activation('relu')(x)
		x = ZeroPadding2D((1, 1))(x)
		x = Conv2D(512, (3, 3), name="block5_conv2")(x)
	layers.append("block5_conv2")
	if (not layer in layers):
		x = Activation('relu')(x)
		x = ZeroPadding2D((1, 1))(x)
		x = Conv2D(512, (3, 3), name="block5_conv3")(x)
	layers.append("block5_conv3")
	if (not layer in layers):
		x = Activation('relu')(x)
		x, pos5 = MaxPooling2D(pool_size=2, strides=2, name="block5_pool")(x)
	layers.append("block5_pool")
	if (not layer in layers):
		x = Flatten(name="flatten")(x)
		x = Dense(4096, name="fc1")(x)
		x = Activation('relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(4096, name="fc2")(x)
		x = Activation('relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(noutputs, activation='softmax', name="predictions")(x)

	if (deconv):
		outputs = [x]
		layers = ['block1_conv1', 'block1_conv2']
		if (not layer in layers):
			outputs.append(pos1)
		layers += ['block1_pool', 'block2_conv1', "block2_conv2"]
		if (not layer in layers):
			outputs.append(pos2)
		layers += ['block2_pool', 'block3_conv1', 'block3_conv2', 'block3_conv3']
		if (not layer in layers):
			outputs.append(pos3)
		layers += ['block3_pool', 'block4_conv1', 'block4_conv2', 'block4_conv3']
		if (not layer in layers):
			outputs.append(pos4)
		layers += ['block4_pool', 'block5_conv1', 'block5_conv2', 'block5_conv3']
		if (not layer in layers):
			outputs.append(pos5)
	else:
		outputs = [x]

	model = Model(inputs = inp, outputs = outputs)

	if weights_path:
		model.load_weights(weights_path, by_name = True)
		print(msg)

	return model

## CREDIT: https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/
def Conv2(pretrained=True, weights_path=None, noutputs=num_classes, deconv=False, sz=32, layer=""):
	if (pretrained and not weights_path):
		weights_path = './data/weights/conv2_weights.h5'

	inp = Input(shape = (sz, sz, 3), name="input")
	x = inp
	x = Conv2D(32, (3, 3), padding='same', name="conv1-1")(x)
	layers = ["conv1-1"]
	if (not layer in layers):
		x = Activation('relu')(x)
		x = Conv2D(32, (3, 3), padding='same', name="conv1-2")(x)
	layers.append("conv1-2")
	if (not layer in layers):
		x = Activation('relu')(x)
		x, pos1 = MaxPooling2D(pool_size=2, strides=2, name="pool1")(x)
	layers.append("pool1")
	if (not layer in layers):
		x = Dropout(0.25)(x)
		x = Conv2D(64, (3, 3), padding='same', name="conv2-1")(x)
	layers.append("conv2-1")
	if (not layer in layers):
		x = Activation('relu')(x)	
		x = Conv2D(64, (3, 3), padding='same', name="conv2-2")(x)
	layers.append("conv2-2")
	if (not layer in layers):
		x = Activation('relu')(x)
		x, pos2 = MaxPooling2D(pool_size=2, strides=2, name="pool2")(x)
	layers.append("pool2")
	if (not layer in layers):
		x = Dropout(0.25)(x)
		x = Flatten()(x)
		x = Dense(512, name="dense1")(x)
		x = Activation('relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(noutputs, activation="softmax", name="dense2")(x)

	if (deconv):
		outputs = [x]
		layers = ['conv1-1', 'conv1-2']
		if (not layer in layers):
			outputs.append(pos1)
		layers += ['pool1', 'conv2-1', "conv2-2"]
		if (not layer in layers):
			outputs.append(pos2)
	else:
		outputs = [x]

	model = Model(inputs = inp, outputs = outputs)

	if weights_path:
		model.load_weights(weights_path, by_name=True)
		print(msg)

	return model

## CREDIT: https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/
def Conv(pretrained=True, weights_path=None, noutputs=num_classes, deconv=False, sz=32, layer=""):
	if (pretrained and not weights_path):
		weights_path = './data/weights/conv_weights.h5'

	inp = Input(shape = (sz, sz, 3), name="input")
	x = inp
	x = Conv2D(32, (3, 3), padding='same', name="conv1")(x)
	layers = ["conv1"]
	if (not layer in layers):
		x = Activation('relu')(x)
		x = Dropout(0.2)(x)
		x = Conv2D(32, (3, 3), padding='same', name="conv2")(x)
	layers.append("conv2")
	if (not layer in layers):
		x = Activation('relu')(x)
		x, pos1 = MaxPooling2D(pool_size=2, strides=2, name="pool2")(x)
	layers.append("pool2")
	if (not layer in layers):
		x = Conv2D(64, (3, 3), padding='same', name="conv3")(x)
	layers.append("conv3")
	if (not layer in layers):
		x = Activation('relu')(x)
		x = Dropout(0.2)(x)
		x = Conv2D(64, (3, 3), padding='same', name="conv4")(x)
	layers.append("conv4")
	if (not layer in layers):
		x = Activation('relu')(x)
		x, pos2 = MaxPooling2D(pool_size=2, strides=2, name="pool4")(x)
	layers.append("pool4")
	if (not layer in layers):
		x = Conv2D(128,(3,3),padding='same',name="conv5")(x)
	layers.append("conv5")
	if (not layer in layers):
		x = Activation('relu')(x)
		x = Dropout(0.2)(x)
		x = Conv2D(128,(3,3),padding='same', name="conv6")(x)
	layers.append("conv6")
	if (not layer in layers):
		x = Activation('relu')(x)
		x, pos3 = MaxPooling2D(pool_size=2, strides=2, name="pool6")(x)
	layers.append("pool6")
	if (not layer in layers):
		x = Flatten()(x)
		x = Dropout(0.2)(x)
		x = Dense(1024, kernel_constraint=maxnorm(3), name="dense1")(x)
		x = Activation('relu')(x)
		x = Dropout(0.2)(x)
		x = Dense(noutputs, activation='softmax', name="dense2")(x)

	if (deconv):
		outputs = [x]
		layers = ['conv1', 'conv2']
		if (not layer in layers):
			outputs.append(pos1)
		layers += ['pool2', 'conv3', 'conv4']
		if (not layer in layers):
			outputs.append(pos2)
		layers += ['pool4', 'conv5', 'conv6']
		if (not layer in layers):
			outputs.append(pos3)
	else:
		outputs = [x]

	model = Model(inputs = inp, outputs = outputs)

	if weights_path:
		model.load_weights(weights_path, by_name=True)
		print(msg)

	return model

## CREDIT: Keras training on CIFAR-10 
## https://gist.github.com/giuseppebonaccorso/e77e505fc7b61983f7b42dc1250f31c8
def Vonc(pretrained=True, weights_path=None, noutputs=num_classes, deconv=False, sz=32, layer=""):
	if (pretrained and not weights_path):
		weights_path = './data/weights/vonc_weights.h5'

	inp = Input(shape = (sz, sz, 3), name="input")
	x = inp
	x = Conv2D(32, (3, 3), name="block1_conv1")(x)
	layers = ["block1_conv1"]
	if (not layer in layers):
		x = Activation('relu')(x)
		x = Conv2D(64, (3, 3), name="block1_conv2")(x)
	layers.append("block1_conv2")
	if (not layer in layers):
		x = Activation('relu')(x)
		x, pos1 = MaxPooling2D(pool_size=2, strides=2, name="pool1")(x)
	layers.append("pool1")
	if (not layer in layers):
		x = Dropout(0.25)(x)
		x = Conv2D(128, (3, 3), name="block2_conv1")(x)
	layers.append("block2_conv1")
	if (not layer in layers):
		x = Activation('relu')(x)
		x, pos2 = MaxPooling2D(pool_size=2, strides=2, name="pool2")(x)
	layers.append("pool2")
	if (not layer in layers):
		x = Conv2D(128, (3, 3), name="block2_conv2")(x)
	layers.append("block2_conv2")
	if (not layer in layers):
		x = Activation('relu')(x)
		x, pos3 = MaxPooling2D(pool_size=2, strides=2, name="pool3")(x)
	layers.append("pool3")
	if (not layer in layers):
		x = Dropout(0.25)(x)
		x = Flatten()(x)
		x = Dense(1024, name="dense1")(x)
		x = Activation('relu')(x)
		x = Dropout(0.25)(x)
		x = Dense(noutputs, activation='softmax', name="dense2")(x)

	if (deconv):
		outputs = [x]
		layers = ['block1_conv1', 'block1_conv2']
		if (not layer in layers):
			outputs.append(pos1)
		layers += ['pool1', 'block2_conv1']
		if (not layer in layers):
			outputs.append(pos2)
		layers += ['pool2', 'block2_conv2']
		if (not layer in layers):
			outputs.append(pos3)
	else:
		outputs = [x]

	model = Model(inputs = inp, outputs = outputs)

	if weights_path:
		model.load_weights(weights_path, by_name=True)
		print(msg)

	return model
