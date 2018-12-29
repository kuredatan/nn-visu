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
import models
import deconv_models
import argparse
import cv2

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
args = parser.parse_args()

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

## Decomment to print 10 random images from X_train
#print_images(X_train, Y_train, num_classes=10, nrows=2)

num_classes = 1000

## Preprocessing
## CREDIT: https://keras.io/preprocessing/image/
X_test = X_test.astype('float32')
X_train = X_train.astype('float32')
X_train /= 255
X_test /= 255
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

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
model = d_models[args.tmodel](pretrained=(args.trun=="testing" and args.trained))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

## "Deconvoluted" version of NN model
deconv_model = d_dmodels[args.tmodel](pretrained=(args.trun=="deconv" and args.trained),
	layer=args.layer if (args.trun=="deconv") else None)
deconv_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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
		preprocessing_function=normalize_input,
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
		preprocessing_function=normalize_input,
	)
	datagen_test.fit(X_test)
	scores = model.evaluate_generator(datagen_test.flow(X_test, Y_test, batch_size=args.batch),
		steps=np.shape(X_test)[0]//args.batch,verbose=2)
	print('Test loss: %.2f' % scores[0])
	print('Test accuracy: %.2f' % scores[1])
if (args.trun == "deconv"):
	im_name = "./data/cats/cat1.jpg"
	im = normalize_input(im_name, sz)
	out = model.predict([im])
	print("Predicted class = " + str(np.argmax(x)))
	backward_net = backward_model()
	out = deconv_model.predict(out)
	## TODO save deconvolved features maps
