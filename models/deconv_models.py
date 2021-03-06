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
from resize import Interp

sz = 32
num_classes = 1000
msg = "* Loaded weights! (DeconvNet)"

# The Deconv2D layers should have the same name as the associated Conv2D layers.
# The shapes can be extracted from [model to deconvolve].summary().

## /!\ No DeconvNet implemented for ResNet_50! 

def ResNet_50(pretrained=True, weights_path=None, noutputs=num_classes, layer="", sz=sz):
	if (pretrained and not weights_path):
		weights_path = './data/weights/resnet50_weights.h5'
	from keras.applications.resnet50 import ResNet50
	model = ResNet50(include_top=True, weights=None, classes=num_classes)
	if weights_path:
		print(msg)
		model.load_weights(weights_path, by_name = True)

	return model

## CREDIT: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
def VGG_16(pretrained=True, weights_path=None, noutputs=num_classes, layer="", sz=sz):
	if (pretrained and not weights_path):
		weights_path = './data/weights/vgg16_weights.h5'

	pos1, pos2, pos3, pos4, pos5 = [None]*5

	if (not layer):
		layer = "block5_pool"

	layers = ["block5_pool"]
	if (layer in layers):
		inp = Input(batch_shape = (1, sz // 32, sz // 32, 128*2*2), name="input")
		x = inp
		pos5 = Input(batch_shape = (1, sz // 32, sz // 32, 128*2*2), name="pos5")
		x = UndoMaxPooling2D((1, sz // 16, sz // 16, 128*2*2), name="block5_pool")([x, pos5])
	layers.append("block5_conv3")
	if (layer in layers):
		if (layer == "block5_conv3"):
			inp = Input(batch_shape = (1, sz // 16, sz // 16, 128*2*2), name="input")
			x = inp
		x = Deconv2D(512,3,padding='SAME',activation='relu', name="block5_conv3")(x)
	layers.append("block5_conv2")
	if (layer in layers):
		if (layer == "block5_conv2"):
			inp = Input(batch_shape = (1, sz // 16, sz // 16, 128*2*2), name="input")
			x = inp
		x = Deconv2D(512,3,padding='SAME',activation='relu', name="block5_conv2")(x)
	layers.append("block5_conv1")
	if (layer in layers):
		if (layer == "block5_conv1"):
			inp = Input(batch_shape = (1, sz // 16, sz // 16, 128*2*2), name="input")
			x = inp
		x = Deconv2D(512,3,padding='SAME',activation='relu', name="block5_conv1")(x)
	layers.append("block4_pool")
	if (layer in layers):
		if (layer == "block4_pool"):
			inp = Input(batch_shape = (1, sz // 16, sz // 16, 128*2*2), name="input")
			x = inp
		pos4 = Input(batch_shape = (1, sz // 16, sz // 16, 128*2*2), name="pos4")
		x = UndoMaxPooling2D((1, sz // 8, sz // 8, 128*2*2), name="block4_pool")([x, pos4])
	layers.append("block4_conv3")
	if (layer in layers):
		if (layer == "block4_conv3"):
			inp = Input(batch_shape = (1, sz // 8, sz // 8, 128*2*2), name="input")
			x = inp
		x = Deconv2D(512,3,padding='SAME',activation='relu', name="block4_conv3")(x)
	layers.append("block4_conv2")
	if (layer in layers):
		if (layer == "block4_conv2"):
			inp = Input(batch_shape = (1, sz // 8, sz // 8, 128*2*2), name="input")
			x = inp
		x = Deconv2D(256*2,3,padding='SAME',activation='relu', name="block4_conv2")(x)
	layers.append("block4_conv1")
	if (layer in layers):
		if (layer == "block4_conv1"):
			inp = Input(batch_shape = (1, sz // 8, sz // 8, 128*2*2), name="input")
			x = inp
		x = Deconv2D(256,3,padding='SAME',activation='relu', name="block4_conv1")(x)
	layers.append("block3_pool")
	if (layer in layers):
		if (layer == "block3_pool"):
			inp = Input(batch_shape = (1, sz // 8, sz // 8, 128*2), name="input")
			x = inp
		pos3 = Input(batch_shape = (1, sz // 8, sz // 8, 128*2), name="pos3")
		x = UndoMaxPooling2D((1, sz // 4, sz // 4, 128*2), name="block3_pool")([x, pos3])
	layers.append("block3_conv3")
	if (layer in layers):
		if (layer == "block3_conv3"):
			inp = Input(batch_shape = (1, sz // 4, sz // 4, 128*2), name="input")
			x = inp
		x = Deconv2D(256,3,padding='SAME',activation='relu', name="block3_conv3")(x)
	layers.append("block3_conv2")
	if (layer in layers):
		if (layer == "block3_conv2"):
			inp = Input(batch_shape = (1, sz // 4, sz // 4, 128*2), name="input")
			x = inp
		x = Deconv2D(256,3,padding='SAME',activation='relu', name="block3_conv2")(x)
	layers.append("block3_conv1")
	if (layer in layers):
		if (layer == "block3_conv1"):
			inp = Input(batch_shape = (1, sz // 4, sz // 4, 128*2), name="input")
			x = inp
		x = Deconv2D(256//2,3,padding='SAME',activation='relu', name="block3_conv1")(x)
	layers.append("block2_pool")
	if (layer in layers):
		if (layer == "block2_pool"):
			inp = Input(batch_shape = (1, sz // 4, sz // 4, 128), name="input")
			x = inp
		pos2 = Input(batch_shape = (1, sz // 4, sz // 4, 128), name="pos2")
		x = UndoMaxPooling2D((1, 112, 112, 128), name="block2_pool")([x, pos2])
	layers.append("block2_conv2")
	if (layer in layers):
		if (layer == "block2_conv2"):
			inp = Input(batch_shape = (1, 112, 112, 128), name="input")
			x = inp
		x = Deconv2D(128,3,padding='SAME',activation='relu', name="block2_conv2")(x)
	layers.append("block2_conv1")
	if (layer in layers):
		if (layer == "block2_conv1"):
			inp = Input(batch_shape = (1, 112, 112, 128), name="input")
			x = inp
		x = Deconv2D(128//2,3,padding='SAME',activation='relu', name="block2_conv1")(x)
	layers.append("block1_pool")
	if (layer in layers):
		if (layer == "block1_pool"):
			inp = Input(batch_shape = (1, 112, 112, 64), name="input")
			x = inp
		pos1 = Input(batch_shape = (1, 112, 112, 64), name="pos1")
		x = UndoMaxPooling2D((1, sz, sz, 64), name="block1_pool")([x, pos1])
	layers.append("block1_conv2")
	if (layer in layers):
		if (layer == "block1_conv2"):
			inp = Input(batch_shape = (1, sz, sz, 64), name="input")
			x = inp
		x = Deconv2D(64,3,padding='SAME',activation='relu', name="block1_conv2")(x)
	layers.append("block1_conv1")
	if (layer in layers):
		if (layer == "block1_conv1"):
			inp = Input(batch_shape = (1, sz, sz, 64), name="input")
			x = inp
		x = Deconv2D(3,3,padding='SAME',activation='relu', name="block1_conv1")(x)

	inputs = [inp]
	for p in [pos1, pos2, pos3, pos4, pos5]:
		if (p != None):
			inputs.append(p)

	model = Model(inputs = inputs, outputs = x)

	if weights_path:
		print(msg)
		model.load_weights(weights_path, by_name = True)

	return model

## CREDIT: https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/
def Conv2(pretrained=True, weights_path=None, noutputs=num_classes, layer="", sz=sz):
	if (pretrained and not weights_path):
		weights_path = './data/weights/conv2_weights.h5'

	pos1, pos2, pos3 = [None]*3

	if (not layer):
		layer = "pool2"

	layers = ["pool2"]
	if (layer in layers):
		inp = Input(batch_shape = (1, sz // 4, sz // 4, 64), name="input")
		x = inp
		pos2 = Input(batch_shape = (1, sz // 4, sz // 4, 64), name="pos2")
		x = UndoMaxPooling2D((1, sz // 2, sz // 2, 64), name="pool2")([x, pos2])
	layers.append("conv2-2")
	if (layer in layers):
		if (layer == "conv2-2"):
			inp = Input(batch_shape = (1, sz // 2, sz // 2, 64), name="input")
			x = inp
		x = Deconv2D(64,3,padding='SAME',activation='relu', name="conv2-2")(x)
	layers.append("conv2-1")
	if (layer in layers):
		if (layer == "conv2-1"):
			inp = Input(batch_shape = (1, sz // 2, sz // 2, 64), name="input")
			x = inp
		x = Deconv2D(64//2,3,padding='SAME',activation='relu', name="conv2-1")(x)
	layers.append("pool1")
	if (layer in layers):
		if (layer == "pool1"):
			inp = Input(batch_shape = (1, sz // 2, sz // 2, 32), name="input")
			x = inp
		pos1 = Input(batch_shape = (1, sz // 2, sz // 2, 32), name="pos1")
		## "Deconvolution" of Dropout layers
		x = Interp((16, 16))(x)
		x = UndoMaxPooling2D((1, sz, sz, 32), name="pool1")([x, pos1])
	layers.append("conv1-2")
	if (layer in layers):
		if (layer == "conv1-2"):
			inp = Input(batch_shape = (1, sz, sz, 32), name="input")
			x = inp
		x = Deconv2D(32, 3, padding='SAME', activation="relu", name="conv1-2")(x)
	layers.append("conv1-1")
	if (layer in layers):
		if (layer == "conv1-1"):
			inp = Input(batch_shape = (1, sz, sz, 32), name="input")
			x = inp
		x = Deconv2D(3, 3, padding='SAME', activation="relu", name="conv1-1")(x)

	inputs = [inp]
	for p in [pos1, pos2, pos3]:
		if (p != None):
			inputs.append(p)

	model = Model(inputs = inputs, outputs = x)

	if weights_path:
		model.load_weights(weights_path, by_name=True)
		print(msg)

	return model

## CREDIT: https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/
def Conv(pretrained=True, weights_path=None, noutputs=num_classes, layer="", sz=sz):
	if (pretrained and not weights_path):
		weights_path = './data/weights/conv_weights.h5'

	pos1, pos2, pos3 = [None]*3

	if (not layer):
		layer = "pool6"

	layers = ["pool6"]
	if (layer in layers):
		inp = Input(batch_shape = (1, sz // 8, sz // 8, 128), name="input")
		x = inp
		pos3 = Input(batch_shape = (1, sz // 8, sz // 8, 128), name="pos3")
		x = UndoMaxPooling2D((1, sz // 4, sz // 4, 128), name="pool6")([x, pos3])
	layers.append("conv6")
	if (layer in layers):
		if (layer == "conv6"):
			inp = Input(batch_shape = (1, sz // 4, sz // 4, 128), name="input")
			x = inp
		x = Deconv2D(128,3,padding='SAME',activation='relu', name="conv6")(x)
	layers.append("conv5")
	if (layer in layers):
		if (layer == "conv5"):
			inp = Input(batch_shape = (1, sz // 4, sz // 4, 128), name="input")
			x = inp	
		x = Deconv2D(128//2,3,padding='SAME',activation='relu', name="conv5")(x)
	layers.append("pool4")
	if (layer in layers):
		if (layer == "pool4"):
			inp = Input(batch_shape = (1, sz // 4, sz // 4, 64), name="input")
			x = inp				
		pos2 = Input(batch_shape = (1, sz // 4, sz // 4, 64), name="pos2")
		x = UndoMaxPooling2D((1, sz // 2, sz // 2, 64), name="pool4")([x, pos2])
	layers.append("conv4")
	if (layer in layers):
		if (layer == "conv4"):
			inp = Input(batch_shape = (1, sz // 2, sz // 2, 64), name="input")
			x = inp
		x = Deconv2D(64, 3, padding='SAME', activation="relu", name="conv4")(x)
	layers.append("conv3")
	if (layer in layers):
		if (layer == "conv3"):
			inp = Input(batch_shape = (1, sz // 2, sz // 2, 64), name="input")
			x = inp
		x = Deconv2D(64//2, 3, padding='SAME', activation="relu", name="conv3")(x)
	layers.append("pool2")
	if (layer in layers):
		if (layer == "pool2"):
			inp = Input(batch_shape = (1, sz // 2, sz // 2, 32), name="input")
			x = inp
		pos1 = Input(batch_shape = (1, sz // 2, sz // 2, 32), name="pos1")
		x = UndoMaxPooling2D((1, sz, sz, 32), name="pool2")([x, pos1])	
	layers.append("conv2")
	if (layer in layers):
		if (layer == "conv2"):
			inp = Input(batch_shape = (1, sz, sz, 32), name="input")
			x = inp
		x = Deconv2D(32, 3, padding='SAME', activation="relu", name="conv2")(x)
	layers.append("conv1")
	if (layer in layers):
		if (layer == "conv1"):
			inp = Input(batch_shape = (1, sz, sz, 32), name="input")
			x = inp
		x = Deconv2D(3, 3, padding='SAME', activation='relu', name="conv1")(x)

	inputs = [inp]
	for p in [pos1, pos2, pos3]:
		if (p != None):
			inputs.append(p)

	model = Model(inputs = inputs, outputs = x)

	if weights_path:
		model.load_weights(weights_path, by_name=True)
		print(msg)

	return model

## CREDIT: Keras training on CIFAR-10 
## https://gist.github.com/giuseppebonaccorso/e77e505fc7b61983f7b42dc1250f31c8
def Vonc(pretrained=True, weights_path=None, noutputs=num_classes, deconv=False, sz=sz, layer=""):
	if (pretrained and not weights_path):
		weights_path = './data/weights/vonc_weights.h5'

	pos1, pos2, pos3 = [None]*3

	if (not layer):
		layer = "pool3"

	layers = ["pool3"]
	if (layer in layers):
		inp = Input(batch_shape = (1, sz // 16, sz // 16, 128), name="input")
		x = inp
		pos3 = Input(batch_shape = (1, sz // 16, sz // 16, 128), name="pos3")
		x = UndoMaxPooling2D((1, sz // 8, sz // 8, 128), name="pool3")([x, pos3])
	layers.append("block2_conv2")
	if (layer in layers):
		if (layer == "block2_conv2"):
			inp = Input(batch_shape = (1, sz // 8, sz // 8, 128), name="input")
			x = inp
		x = Deconv2D(128,3,padding='SAME',activation='relu', name="block2_conv2")(x)
	layers.append("pool2")
	if (layer in layers):
		if (layer == "pool2"):
			inp = Input(batch_shape = (1, 6, 6, 128), name="input")
			x = inp
		## "Deconvolution" of Dropout layers
		x = Interp((6, 6))(x)
		pos2 = Input(batch_shape = (1, 6, 6, 128), name="pos2")
		x = UndoMaxPooling2D((1, 12, 12, 128), name="pool2")([x, pos2])
	layers.append("block2_conv1")
	if (layer in layers):
		if (layer == "block2_conv1"):
			inp = Input(batch_shape = (1, 12, 12, 128), name="input")
			x = inp
		## "Deconvolution" of Dropout layers
		x = Interp((14, 14))(x)
		x = Deconv2D(128//2,3,padding='SAME',activation='relu', name="block2_conv1")(x)
	layers.append("pool1")
	if (layer in layers):
		if (layer == "pool1"):
			inp = Input(batch_shape = (1, 14, 14, 64), name="input")
			x = inp
		pos1 = Input(batch_shape = (1, 14, 14, 64), name="pos1")
		x = UndoMaxPooling2D((1, 28, 28, 64), name="pool1")([x, pos1])
	layers.append("block1_conv2")
	if (layer in layers):
		if (layer == "block1_conv2"):
			inp = Input(batch_shape = (1, 28, 28, 64), name="input")
			x = inp
		x = Deconv2D(64//2, 3, padding='SAME', activation="relu", name="block1_conv2")(x)
	layers.append("block1_conv1")
	if (layer in layers):
		if (layer == "block1_conv1"):
			inp = Input(batch_shape = (1, 30, 30, 32), name="input")
			x = inp
		x = Deconv2D(3, 3, padding='SAME', activation='relu', name="block1_conv1")(x)
	
	inputs = [inp]

	for p in [pos1, pos2, pos3]:
		if (p != None):
			inputs.append(p)

	model = Model(inputs = inputs, outputs = x)

	if weights_path:
		model.load_weights(weights_path, by_name=True)
		print(msg)

	return model


