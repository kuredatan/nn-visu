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

from keras.applications.vgg16 import VGG16

## Get weights for VGG16
#model = VGG16(include_top=True, weights='imagenet', classes=num_classes)
#model.save_weights('vgg16_weights.h5')

def VGG_16(pretrained=True, weights_path=None, noutputs=num_classes, deconv=False, sz=32):
	if (weights_path):
		print("* Loaded weights!")
		weights = "imagenet"
	else:
		weights = None
	model = VGG16(include_top=True, weights=weights, input_shape=(sz, sz, 3), classes=1000)
	# https://stackoverflow.com/questions/41668813/how-to-add-and-remove-new-layers-in-keras-after-loading-weights
	if (num_classes != 1000):
		model.layers.pop()
		x = Dense(num_classes, activation='softmax', name="predictions")(model.layers[-1].output)
		model = Model(input=model.input, output=[x])
	return model

## CREDIT: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
def VGG_16_2(pretrained=True, weights_path=None, noutputs=num_classes, deconv=False, sz=32):
	if (pretrained):
		weights_path = './data/weights/vgg16_weights.h5'

	inp = Input(shape = (sz, sz, 3), name="input_1")

	try:
		x = Lambda(lambda image: ktf.image.resize_images(image, (224, 224)))(inp)
	except :
		# if you have older version of tensorflow
		x = Lambda(lambda image: ktf.image.resize_images(image, 224, 224))(inp)

	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(64, (3, 3), activation='relu', name="block1_conv1")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(64, (3, 3), activation='relu', name="block1_conv2")(x)
	x, pos1 = MaxPooling2D(pool_size=2, strides=2, name="block1_pool")(x)

	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(128, (3, 3), activation='relu', name="block2_conv1")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(128, (3, 3), activation='relu', name="block2_conv2")(x)
	x, pos2 = MaxPooling2D(pool_size=2, strides=2, name="block2_pool")(x)

	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(256, (3, 3), activation='relu', name="block3_conv1")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(256, (3, 3), activation='relu', name="block3_conv2")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(256, (3, 3), activation='relu', name="block3_conv3 ")(x)
	x, pos3 = MaxPooling2D(pool_size=2, strides=2, name="block3_pool")(x)
	
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(256, (3, 3), activation='relu', name="block4_conv1")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(256, (3, 3), activation='relu', name="block4_conv2")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(256, (3, 3), activation='relu', name="block4_conv3")(x)
	x, pos4 = MaxPooling2D(pool_size=2, strides=2, name="block4_pool")(x)

	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(512, (3, 3), activation='relu', name="block5_conv1")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(512, (3, 3), activation='relu', name="block5_conv2")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = Conv2D(512, (3, 3), activation='relu', name="block5_conv3")(x)
	x, pos5 = MaxPooling2D(pool_size=2, strides=2, name="block5_pool")(x)

	x = Flatten(name="flatten")(x)
	x = Dense(4096, activation='relu', name="fc1")(x)
	x = Dropout(0.5)(x)
	x = Dense(4096, activation='relu', name="fc2")(x)
	x = Dropout(0.5)(x)
	x = Dense(noutputs, activation='softmax', name="predictions")(x)

	if (deconv):
		outputs = [x, pos1, pos2, pos3, pos4, pos5]
	else:
		outputs = [x]

	model = Model(inputs = inp, outputs = outputs)

	if weights_path:
		print("* Loaded weights!")
		model.load_weights(weights_path, by_name = True)

	return model

## CREDIT: https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/
def Conv2(pretrained=True, weights_path=None, noutputs=num_classes, deconv=False, sz=32):
	if (pretrained):
		weights_path = './data/weights/conv2_weights.h5'

	inp = Input(shape = (sz, sz, 3))
	x = inp

	x = Conv2D(32, (3, 3), padding='same', activation="relu", name="conv1-1")(x)
	x = Conv2D(32, (3, 3), padding='same', activation="relu", name="conv1-2")(x)
	x, pos1 = MaxPooling2D(pool_size=2, strides=2, name="pool1")(x)
	x = Dropout(0.25)(x)

	x = Conv2D(64, (3, 3), padding='same', activation="relu", name="conv2-1")(x)
	x = Conv2D(64, (3, 3), padding='same', activation="relu", name="conv2-2")(x)
	x, pos2 = MaxPooling2D(pool_size=2, strides=2, name="pool2")(x)
	x = Dropout(0.25)(x)

	x = Flatten()(x)
	x = Dense(512, name="dense1", activation="relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(noutputs, activation="softmax", name="dense2")(x)

	if (deconv):
		outputs = [x, pos1, pos2]
	else:
		outputs = [x]

	model = Model(inputs = inp, outputs = outputs)

	if weights_path:
		print("* Loaded weights!")
		model.load_weights(weights_path, by_name=True)

	return model

## CREDIT: https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/
def Conv(pretrained=True, weights_path=None, noutputs=num_classes, deconv=False, sz=32):
	if (pretrained):
		weights_path = './data/weights/conv_weights.h5'

	inp = Input(shape = (sz, sz, 3))
	x = inp

	x = Conv2D(32, (3, 3), padding='same', activation='relu', name="conv1")(x)
	x = Dropout(0.2)(x)

	x = Conv2D(32, (3, 3), padding='same', activation="relu", name="conv2")(x)
	x, pos1 = MaxPooling2D(pool_size=2, strides=2, name="pool2")(x)

	x = Conv2D(64, (3, 3), padding='same', activation="relu", name="conv3")(x)
	x = Dropout(0.2)(x)

	x = Conv2D(64, (3, 3), padding='same', activation="relu", name="conv4")(x)
	x, pos2 = MaxPooling2D(pool_size=2, strides=2, name="pool4")(x)

	x = Conv2D(128,(3,3),padding='same',activation='relu', name="conv5")(x)
	x = Dropout(0.2)(x)

	x = Conv2D(128,(3,3),padding='same',activation='relu', name="conv6")(x)
	x, pos3 = MaxPooling2D(pool_size=2, strides=2, name="pool6")(x)

	x = Flatten()(x)
	x = Dropout(0.2)(x)
	x = Dense(1024,activation='relu',kernel_constraint=maxnorm(3), name="dense1")(x)
	x = Dropout(0.2)(x)
	x = Dense(noutputs, activation='softmax', name="dense2")(x)

	if (deconv):
		outputs = [x, pos1, pos2, pos3]
	else:
		outputs = [x]

	model = Model(inputs = inp, outputs = outputs)

	if weights_path:
		print("* Loaded weights!")
		model.load_weights(weights_path, by_name=True)

	return model

## CREDIT: Keras training on CIFAR-10 
## https://gist.github.com/giuseppebonaccorso/e77e505fc7b61983f7b42dc1250f31c8
def Vonc(pretrained=True, weights_path=None, noutputs=num_classes, deconv=False, sz=32):
	if (pretrained):
		weights_path = './data/weights/vonc_weights.h5'

	inp = Input(shape = (sz, sz, 3))
	x = inp

	x = Conv2D(32, (3, 3), activation='relu', name="block1_conv1")(x)
	x = Conv2D(64, (3, 3), activation='relu', name="block1_conv2")(x)
	x, pos1 = MaxPooling2D(pool_size=2, strides=2, name="pool1")(x)
	x = Dropout(0.25)(x)

	x = Conv2D(128, (3, 3), activation='relu', name="block2_conv1")(x)
	x, pos2 = MaxPooling2D(pool_size=2, strides=2, name="pool2")(x)
	x = Conv2D(128, (3, 3), activation='relu', name="block2_conv2")(x)
	x, pos3 = MaxPooling2D(pool_size=2, strides=2, name="pool3")(x)
	x = Dropout(0.25)(x)

	x = Flatten()(x)
	x = Dense(1024,activation='relu',name="dense1")(x)
	x = Dropout(0.25)(x)
	x = Dense(noutputs, activation='softmax', name="dense2")(x)

	if (deconv):
		outputs = [x, pos1, pos2, pos3]
	else:
		outputs = [x]

	model = Model(inputs = inp, outputs = outputs)

	if weights_path:
		print("* Loaded weights!")
		model.load_weights(weights_path, by_name=True)

	return model
