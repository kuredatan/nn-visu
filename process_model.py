#coding: utf-8

from __future__ import print_function
import sys
sys.path += ['./layers/', './utils/', './models/']

import numpy as np
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from random import sample
from copy import deepcopy
from print_norm_utils import print_images, plot_kernels, normalize_input
from KerasDeconv import DeconvNet
import cPickle as pickle
import models
import deconv_models
import argparse
import glob
import cv2
import os
from utils import get_deconv_images, plot_deconv, plot_max_activation, find_top9_mean_act

## CREDIT: Adapted from practicals/HMW2 from Andrea Vedaldi and Andrew Zisserman 
## by Gul Varol and Ignacio Rocco in PyTorch

## Training is performed on training set of CIFAR-10
## Testing is performed either on testing set of CIFAR-10, either on another dataset

parser = argparse.ArgumentParser(description='Simple NN models')
parser.add_argument('--tmodel', type=str, default='conv', metavar='M',
                    help='In ["conv2", "conv", "vgg"].')
parser.add_argument('--trun', type=str, default='training', metavar='R',
                    help='In ["training", "testing", "deconv"].')
parser.add_argument('--tdata', type=str, default='CIFAR-10', metavar='D',
                    help='In ["CIFAR-10", "CATS"].')
parser.add_argument('--batch', type=int, default=64, metavar='B',
                    help='Batch size.')
parser.add_argument('--epoch', type=int, default=10, metavar='E',
                    help='Number of epochs.')
parser.add_argument('--optimizer', type=str, default="SGD", metavar='O',
                    help='SGD/Adam optimizers.')
parser.add_argument('--trained', type=int, default=1, metavar='T',
                    help='Import initialized weights [0, 1].')
parser.add_argument('--lr', type=float, default=0.1, metavar='L',
                    help='Learning rate in (0,1).')
parser.add_argument('--decay', type=float, default=1e-6, metavar='C',
                    help='Decay rate in (0,1).')
parser.add_argument('--momentum', type=float, default=0.9, metavar='U',
                    help='Momentum.')
parser.add_argument('--layer', type=str, default="conv1-1", metavar='Y',
                    help='Name of the layer to deconvolve.')
parser.add_argument('--verbose', type=int, default=0, metavar='V',
                    help='Whether to print things or not: in [0, 1].')
args = parser.parse_args()

folder = "./data/figures/"
if not os.path.exists(folder):
	os.mkdir(folder)

folder += args.tdata + "/"
if not os.path.exists(folder):
	os.mkdir(folder)

if (args.optimizer == "SGD"):
	optimizer = SGD(lr = args.lr, decay=args.decay, momentum=args.momentum, nesterov=True)
if (args.optimizer == "Adam"):
	optimizer = Adam(lr=args.lr, decay=args.decay)

## CREDIT: Keras training on CIFAR-10 
## https://gist.github.com/giuseppebonaccorso/e77e505fc7b61983f7b42dc1250f31c8

# For reproducibility
np.random.seed(1000)

# Load data
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
## CREDIT: https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DeconvNet
if (args.trun != "training" and args.tdata == "CATS"):
	list_img = glob.glob("./data/cats/*.jpg*")
	assert len(list_img) > 0, "Put some images in the ./data/cats folder"
	if len(list_img) < 32:
		list_img = (int(32 / len(list_img)) + 2) * list_img
		list_img = list_img[:32]
	data = np.array([normalize_input(im_name, 32) for im_name in list_img])
	X_test = np.array(data)
	## Source for ImageNet labels: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
	Y_test = [283, 281, 285, 281, 284, 282, 282, 281, 281, 281, 282]

## Decomment to print 10 random images from X_train
#print_images(X_train, Y_train, num_classes=10, nrows=2)

num_classes = 1000

## Preprocessing
## CREDIT: https://keras.io/preprocessing/image/
X_test = X_test.astype('float32')
X_train = X_train.astype('float32')
X_train /= 255.0
X_test /= 255.0
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)
Y_test = Y_test.astype('float32')
Y_train = Y_train.astype('float32')

## Cut X_train into training and validation datasets
p = 0.30
n = np.shape(X_train)[0]
in_val = sample(range(n), int(p*n))
in_train = list(set(in_val).symmetric_difference(range(n)))
X_val = X_train[in_val, :, :, :]
Y_val = Y_train[in_val, :]
X_train = X_train[in_train, :, :, :]
Y_train = Y_train[in_train, :]

d_models = {"conv": models.Conv, "vgg": models.VGG_16, "conv2": models.Conv2}
d_dmodels = {"conv": deconv_models.Conv, "vgg": deconv_models.VGG_16, "conv2": deconv_models.Conv2}

## NN model
model = d_models[args.tmodel](pretrained=(args.trun=="testing" and args.trained), deconv=args.trun == "deconv")
if (args.trun != "deconv"):
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

## "Deconvoluted" version of NN models
#deconv_model = d_dmodels[args.tmodel](pretrained=(args.trun=="deconv" and args.trained), layer=args.layer if (args.trun=="deconv") else None)
#deconv_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
deconv_model = DeconvNet(model)

## Print kernels in a given layer (for instance "conv1-1")
#layers = [layer.name for layer in model.layers]
#plot_kernels(model, layers[0])

###########################################
## TRAINING/TESTING/DECONV PIPELINES     ##
###########################################

if (args.trun == "training"):
	datagen_train = ImageDataGenerator(
		featurewise_center=False,
		featurewise_std_normalization=False,
		## Normalization
		preprocessing_function=lambda x : normalize_input(x, 32),
		## Data augmentation
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
	)
	datagen_val = deepcopy(datagen_train)
	# compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied)
	datagen_train.fit(X_train)
	datagen_val.fit(X_val)
	# fits the model on batches with real-time data augmentation:
	hist = model.fit_generator(datagen_train.flow(X_train, Y_train, batch_size=args.batch),
		shuffle=True,
		steps_per_epoch=int((1-p)*n)//args.batch,
		epochs=args.epoch,
		validation_data=datagen_val.flow(X_val, Y_val),
		validation_steps=int(p*n)//args.batch,
		#callbacks=[EarlyStopping(min_delta=0.001, patience=10)],
		verbose=2)
	print("Final Accuracy: %.3f" % hist.history["acc"][-1])
	print("Final (Validation) Accuracy: %.3f" % hist.history["val_acc"][-1])
	print("Final Loss: %.3f" % hist.history["loss"][-1])
	print("Final (Validation) Loss: %.3f" %hist.history["val_loss"][-1])
	model.save_weights('./data/weights/'+args.tmodel+'_weights.h5')
if (args.trun == "testing"):
	datagen_test = ImageDataGenerator(
		featurewise_center=False,
		featurewise_std_normalization=False,
		## Normalization
		preprocessing_function=lambda x : normalize_input(x, 32),
	)
	datagen_test.fit(X_test)
	scores = model.evaluate_generator(datagen_test.flow(X_test, Y_test, batch_size=args.batch),
		steps=np.shape(X_test)[0]//args.batch,verbose=2)
	if (args.verbose):
		print(model.summary())
	print('Test loss: %.2f' % scores[0])
	print('Test accuracy: %.2f' % scores[1])
if (args.trun == "deconv"):
	i = 0
	out = model.predict([X_test[i]])
	print("Predicted class = " + str(np.argmax(out[0])))
	out = deconv_model.predict(out)
	plt.figure(figsize=(20, 20))
	plt.imshow(out)
	## Save output feature map
	#If you want to reconstruct from a single feature map / activation, you can
	# simply set all the others to 0. (in file "models/deconv_models.py")
	plt.savefig(folder + "fmap_" + str(i) + ".png", bbox_inches="tight")
