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
msg = "* Loaded weights! (DeconvNet)"

#If you want to reconstruct from a single feature map / activation, you can
# simply set all the others to 0.

# The Deconv2D layers should have the same name as the associated Conv2D layers.
# The shapes can be extracted from [model to deconvolve].summary().

## CREDIT: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
def VGG_16(pretrained=True, weights_path=None, noutputs=num_classes, layer="", sz=sz):
	if (pretrained):
		weights_path = './data/weights/vgg16_weights.h5'

	inp = Input(batch_shape = (1, sz // 4, sz // 4, 128*2*2))
	x = inp

	pos5 = Input(batch_shape = (1, sz // 4, sz // 4, 128*2))
	x = UndoMaxPooling2D((1, sz, sz, 128*2), name="block5_pool")([x, pos5])
	x = Deconv2D(512//2,3,padding='SAME',activation='relu', name="block5_conv3")(x)
	x = Deconv2D(512//2,3,padding='SAME',activation='relu', name="block5_conv2")(x)
	x = Deconv2D(512//2,3,padding='SAME',activation='relu', name="block5_conv1")(x)

	pos4 = Input(batch_shape = (1, sz // 4, sz // 4, 128))
	x = UndoMaxPooling2D((1, sz, sz, 128), name="block4_pool")([x, pos4])
	x = Deconv2D(256//2,3,padding='SAME',activation='relu', name="block4_conv3")(x)
	x = Deconv2D(256//2,3,padding='SAME',activation='relu', name="block4_conv2")(x)
	x = Deconv2D(256//2,3,padding='SAME',activation='relu', name="block4_conv1")(x)

	pos3 = Input(batch_shape = (1, sz // 4, sz // 4, 64))
	x = UndoMaxPooling2D((1, sz, sz, 64), name="block3_pool")([x, pos3])
	x = Deconv2D(256//2,3,padding='SAME',activation='relu', name="block3_conv3")(x)
	x = Deconv2D(256//2,3,padding='SAME',activation='relu', name="block3_conv2")(x)
	x = Deconv2D(256//2,3,padding='SAME',activation='relu', name="block3_conv1")(x)

	pos2 = Input(batch_shape = (1, sz // 4, sz // 4, 32))
	x = UndoMaxPooling2D((1, sz, sz, 32), name="block2_pool")([x, pos2])
	x = Deconv2D(128//2,3,padding='SAME',activation='relu', name="block2_conv2")(x)
	x = Deconv2D(128//2,3,padding='SAME',activation='relu', name="block2_conv1")(x)

	pos1 = Input(batch_shape = (1, sz // 4, sz // 4, 16))
	x = UndoMaxPooling2D((1, sz, sz, 16), name="block1_pool")([x, pos1])
	x = Deconv2D(64//2,3,padding='SAME',activation='relu', name="block1_conv2")(x)
	x = Deconv2D(64//2,3,padding='SAME',activation='relu', name="block1_conv1")(x)

	model = Model(inputs = [inp, pos1, pos2, pos3, pos4, pos5], outputs = x)

	if weights_path:
		print(msg)
		model.load_weights(weights_path, by_name = True)

	return model

## CREDIT: https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/
def Conv2(pretrained=True, weights_path=None, noutputs=num_classes, layer="", sz=sz):
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
		print(msg)
		model.load_weights(weights_path, by_name=True)

	return model

## CREDIT: https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/
def Conv(pretrained=True, weights_path=None, noutputs=num_classes, layer="", sz=sz):
	if (pretrained):
		weights_path = './data/weights/conv_weights.h5'

	pos1, pos2, pos3 = [None]*3

	layers = ["pool6"]
	if (layer in layers):
		inp = Input(batch_shape = (1, sz // 8, sz // 8, 128), name="inp")
		x = inp
		pos3 = Input(batch_shape = (1, sz // 8, sz // 8, 128), name="pos3")
		x = UndoMaxPooling2D((1, sz // 4, sz // 4, 128), name="pool6")([x, pos3])
	layers.append("conv6")
	if (layer in layers):
		if (layer == "conv6"):
			inp = Input(batch_shape = (1, sz // 4, sz // 4, 128), name="inp")
			x = inp
		x = Deconv2D(128,3,padding='SAME',activation='relu', name="conv6")(x)
	layers.append("conv5")
	if (layer in layers):
		if (layer == "conv5"):
			inp = Input(batch_shape = (1, sz // 4, sz // 4, 128), name="inp")
			x = inp	
		x = Deconv2D(128//2,3,padding='SAME',activation='relu', name="conv5")(x)
	layers.append("pool4")
	if (layer in layers):
		if (layer == "pool4"):
			inp = Input(batch_shape = (1, sz // 4, sz // 4, 64), name="inp")
			x = inp				
		pos2 = Input(batch_shape = (1, sz // 4, sz // 4, 64), name="pos2")
		x = UndoMaxPooling2D((1, sz // 2, sz // 2, 64), name="pool4")([x, pos2])
	layers.append("conv4")
	if (layer in layers):
		if (layer == "conv4"):
			inp = Input(batch_shape = (1, sz // 2, sz // 2, 64), name="inp")
			x = inp
		x = Deconv2D(64, 3, padding='SAME', activation="relu", name="conv4")(x)
	layers.append("conv3")
	if (layer in layers):
		if (layer == "conv3"):
			inp = Input(batch_shape = (1, sz // 2, sz // 2, 64), name="inp")
			x = inp
		x = Deconv2D(64//2, 3, padding='SAME', activation="relu", name="conv3")(x)
	layers.append("pool2")
	if (layer in layers):
		if (layer == "pool2"):
			inp = Input(batch_shape = (1, sz // 2, sz // 2, 32), name="inp")
			x = inp
		pos1 = Input(batch_shape = (1, sz // 2, sz // 2, 32), name="pos1")
		x = UndoMaxPooling2D((1, sz, sz, 32), name="pool2")([x, pos1])	
	layers.append("conv2")
	if (layer in layers):
		if (layer == "conv2"):
			inp = Input(batch_shape = (1, sz, sz, 32), name="inp")
			x = inp
		x = Deconv2D(32, 3, padding='SAME', activation="relu", name="conv2")(x)
	layers.append("conv1")
	if (layer in layers):
		if (layer == "conv1"):
			inp = Input(batch_shape = (1, sz, sz, 32), name="inp")
			x = inp
		x = Deconv2D(3, 3, padding='SAME', activation='relu', name="conv1")(x)

	inputs = [inp]
	if (pos1 != None):
		inputs.append(pos1)
	if (pos2 != None):
		inputs.append(pos2)
	if (pos3 != None):
		inputs.append(pos3)

	model = Model(inputs = inputs, outputs = x)

	if weights_path:
		print(msg)
		model.load_weights(weights_path, by_name=True)

	return model

## CREDIT: Keras training on CIFAR-10 
## https://gist.github.com/giuseppebonaccorso/e77e505fc7b61983f7b42dc1250f31c8
def Vonc(pretrained=True, weights_path=None, noutputs=num_classes, deconv=False, sz=sz, layer=""):
	if (pretrained):
		weights_path = './data/weights/vonc_weights.h5'

	inp = Input(batch_shape = (1, sz // 4, sz // 4, 128))
	x = inp

	pos3 = Input(batch_shape = (1, sz // 4, sz // 4, 128))
	x = UndoMaxPooling2D((1, sz, sz, 128), name="pool3")([x, pos3])
	x = Deconv2D(128,3,padding='SAME',activation='relu', name="block2_conv2")(x)
	pos2 = Input(batch_shape = (1, sz, sz, 64))
	x = UndoMaxPooling2D((1, sz, sz, 64), name="pool2")([x, pos2])
	x = Deconv2D(128//2,3,padding='SAME',activation='relu', name="block2_conv1")(x)

	pos1 = Input(batch_shape = (1, sz, sz, 32))
	x = UndoMaxPooling2D((1, sz, sz, 32), name="pool1")([x, pos1])
	x = Deconv2D(64//2, 3, padding='SAME', activation="relu", name="block1_conv2")(x)
	x = Deconv2D(3, 3, padding='SAME', activation='relu', name="block1_conv1")(x)

	model = Model(inputs = [inp, pos1, pos2, pos3], outputs = x)

	if weights_path:
		print(msg)
		model.load_weights(weights_path, by_name=True)

	return model


