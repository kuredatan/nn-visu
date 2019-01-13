#coding: utf-8

from __future__ import print_function
import sys
sys.path += ['./layers/', './utils/', './models/']

import numpy as np
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
from random import sample
from copy import deepcopy
from print_norm_utils import print_images, plot_kernels, normalize_input, query_yes_no, load_input, resize
from keras.applications.vgg16 import preprocess_input
import cPickle as pickle
import models
import deconv_models
import argparse
import glob
import cv2
import os
from imagenet1000 import imagenet1000
from cats import dict_labels_cats
from utils import get_deconv_images, plot_deconv, plot_max_activation, find_top9_mean_act
import matplotlib.pyplot as plt

## Training models
# python2.7 process_model.py --tmodel vonc --tdata CIFAR-10 --trun training --trained 0 --epoch 250 --lr 0.01 --optimizer Adam --batch 64
# python2.7 process_model.py --tmodel conv --tdata CIFAR-10 --trun training --trained 0 --epoch 10 --lr 0.0001 --optimizer Adam --batch 128
# python2.7 process_model.py --tmodel conv2 --tdata CIFAR-10 --trun training --trained 0 --epoch 10 --lr 0.01 --optimizer SGD --batch 128
## Testing
# python2.7 process_model.py --tmodel conv --tdata CIFAR-10 --trun testing --trained 1
## Deconvolution
# python2.7 process_model.py --tmodel conv --trained 1 --batch 1 --trun deconv --tlayer conv6 --tdata CATS --verbose 1

## CREDIT: Adapted from practicals/HMW2 from Andrea Vedaldi and Andrew Zisserman 
## by Gul Varol and Ignacio Rocco in PyTorch

## Training is performed on training set of CIFAR-10
## Testing is performed either on testing set of CIFAR-10, either on another dataset

##################################################################################
##################################################################################
############# ARGUMENTS

# For reproducibility
np.random.seed(1000)

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
                    help='Import initialized weights: in {0, 1}.')
parser.add_argument('--lr', type=float, default=0.1, metavar='L',
                    help='Learning rate in (0,1).')
parser.add_argument('--decay', type=float, default=1e-6, metavar='C',
                    help='Decay rate in (0,1).')
parser.add_argument('--momentum', type=float, default=0.9, metavar='U',
                    help='Momentum.')
parser.add_argument('--tlayer', type=str, default="", metavar='Y',
                    help='Name of the layer to deconvolve.')
parser.add_argument('--verbose', type=int, default=0, metavar='V',
                    help='Whether to print things or not: in {0, 1}.')
#parser.add_argument('--subtask', type=str, default="", metavar='S',
#                    help='Sub-task for deconvolution.')
#parser.add_argument('--tdeconv', type=str, default="custom", metavar='K',
#                    help='Choice of implementation for deconvolution: Mihai Dusmanu\'s (\'custom\') or DeepLearningImplementations (\'keras\').')
parser.add_argument('--loss', type=str, default="categorical_crossentropy", metavar='O',
                    help='Choice of loss function, among those supported by Keras.')
args = parser.parse_args()

folder = "./data/figures/"
if not os.path.exists(folder):
	os.mkdir(folder)

folder += args.tdata + "/"
if not os.path.exists(folder):
	os.mkdir(folder)

if not os.path.exists("./Figures/"):
        os.makedirs("./Figures/")
if not os.path.exists("./Figures/"+args.tmodel+"/"):
        os.makedirs("./Figures/"+args.tmodel+"/")

if (args.optimizer == "SGD"):
	optimizer = SGD(lr = args.lr, decay=args.decay, momentum=args.momentum, nesterov=True)
if (args.optimizer == "Adam"):
	optimizer = Adam(lr=args.lr, decay=args.decay)
if (args.optimizer == "rmsprop"):
	optimizer = RMSprop(lr=args.lr)

##################################################################################
##################################################################################
############# DATA

# Load data
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

num_classes = 1000

def print_image(x):
	plt.imshow(np.resize(x, (sz, sz, 3)))
	plt.show()
	raise ValueError

if (args.tmodel == "vgg"):
	sz = 224
	preprocess_image = lambda x : resize(x, (1, sz, sz, 3))
	#preprocess_input(resize(x, (1, sz, sz, 3)))
else:
	sz = 32
	training_means = [np.mean(X_train[:,:,i].astype('float32')) for i in range(3)]
	preprocess_image = lambda x : normalize_input(x, sz, training_means)

## CREDIT: Keras training on CIFAR-10 
## https://gist.github.com/giuseppebonaccorso/e77e505fc7b61983f7b42dc1250f31c8

## CREDIT: https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DeconvNet
if (args.trun != "training" and args.tdata == "CATS"):
	list_img = glob.glob("./data/cats/*.jpg*")
	assert len(list_img) > 0, "Put some images in the ./data/cats folder"
	labels = [dict_labels_cats[img] for img in list_img]
	if len(list_img) < args.batch:
		list_img = (int(args.batch / len(list_img)) + 2) * list_img
		list_img = list_img[:args.batch]
		labels = (int(args.batch / len(labels)) + 2) * labels
		labels = labels[:args.batch]
	data = np.array([load_input(im_name, sz) for im_name in list_img])
	if (len(np.shape(data)) > 4):
		X_test = data.reshape((np.shape(data)[0], np.shape(data)[2], np.shape(data)[3], np.shape(data)[4]))
	else:
		X_test = data
	Y_test = np.array(labels)

## Preprocessing
## CREDIT: https://keras.io/preprocessing/image/
Y_train_c = to_categorical(Y_train, num_classes)
Y_test_c = to_categorical(Y_test, num_classes)
Y_train_c = Y_train_c.astype('float32')
Y_test_c = Y_test_c.astype('float32')

## Decomment to print 5*nrows random images from X_train
#print_images(X_train, Y_train, num_classes=10, nrows=2)
#print_images(X_test, Y_test, num_classes=num_classes, nrows=2)

## Cut X_train into training and validation datasets
p = 0.30
n = np.shape(X_train)[0]
in_val = sample(range(n), int(p*n))
in_train = list(set(in_val).symmetric_difference(range(n)))
X_val = X_train[in_val, :, :, :]
Y_val = Y_train[in_val]
Y_val_c = Y_train_c[in_val, :]
X_train = X_train[in_train, :, :, :]
Y_train_c = Y_train_c[in_train, :]

##################################################################################
##################################################################################
############# MODELS

d_models = {"conv": models.Conv, "vgg": models.VGG_16, "conv2": models.Conv2, "vonc": models.Vonc}
d_dmodels = {"conv": deconv_models.Conv, "vgg": deconv_models.VGG_16, "conv2": deconv_models.Conv2, "vonc": deconv_models.Vonc}

## NN model
model = d_models[args.tmodel](pretrained=args.trained>0, deconv=args.trun == "deconv", sz=sz, layer=args.tlayer)
if (args.trun != "deconv"):
	model.compile(loss=args.loss, optimizer=optimizer, metrics=['accuracy'])

## Print kernels in a given layer
layers = [layer.name for layer in model.layers]
if (args.verbose == 1):
	print("Layer names for model " + args.tmodel + ":")
	print(layers)
	print("______________________\nSummary:")
	print(model.summary())

layers = list(map(lambda x : x.name, model.layers))
#layer = layers[1]
#print("Plotting kernel from layer \'" + layer + "\'")
#plot_kernels(model, layer)

## "Deconvoluted" version of NN models
if (args.trun == "deconv"):
	if (not args.tlayer in layers):
		print(args.tlayer + " is not in layer list: " + str(layers))
		raise ValueError
	deconv_model = d_dmodels[args.tmodel](pretrained=args.trained>0, layer=args.tlayer, sz=sz)
	deconv_model.compile(loss=args.loss, optimizer=optimizer, metrics=['accuracy'])
## Or the implementation of DeconvNet in Keras
#deconv_model = DeconvNet(model)

###########################################
## TRAINING/TESTING/DECONV PIPELINES     ##
###########################################

def run_nn(datagen, X, Y_c, Y, batch_size, training=False, verbose=True, kmin=10):
	datagen.fit(X)
	labels = []
	n = np.shape(X)[0]
	epochs = args.epoch if (training) else 1
	Y_test = []
	batch = 0
	if (training):
		hist = model.fit_generator(datagen.flow(X, Y_c, batch_size=batch_size), verbose=1,
			epochs=epochs,shuffle=True,steps_per_epoch=np.shape(X)[0]//batch_size, 
			callbacks=[EarlyStopping(monitor="loss", min_delta=0.001, patience=3)])
	else:
		for x_batch, y_batch in datagen.flow(X, Y_c, batch_size=batch_size):
			if (args.verbose):
				print("Batch #" + str(batch+1) + "/" + str(n/batch_size+1))
			predictions = model.predict(x_batch, batch_size, verbose=1)
			try:
				pred_ = [np.argmax(predictions[0][i, :]) for i in range(len(predictions))]
			except:
				pred_ = [np.argmax(predictions[i, :]) for i in range(len(predictions))]
			labels += pred_
			Y_batch = [np.argmax(y_batch[i, :]) for i in range(len(predictions))]
			Y_test += Y_batch
			acc = np.array(pred_)==np.array(Y_batch)
			acc = np.sum(acc)/float(len(predictions))
			if (args.verbose):
				print(str(batch_size*(batch+1)) + "/" + str(n) + ": acc = " + str(acc))
			if batch >= n / batch_size:
				break
			else:
				batch += 1
	if (not training):
		if (verbose):
			acc = np.sum(np.array(labels) == np.array(Y_test))/float(len(labels))
			if (args.verbose):
				print(model.summary())
			k = min(np.shape(labels)[0], kmin)
			pred = [imagenet1000[label] for label in labels]
			if (args.tdata == "CATS"):
				real = [imagenet1000[y] for y in Y_test]
			else:
				real = [imagenet1000[y] for y in Y_test]
			print("")
			print("PREDICTED" + "\t"*3 + "REAL LABELS")
			for i in range(k):
				print(pred[i] + " - " + real[i])
			print('')
			print('* ACCURACY %.2f' % acc)
		return labels
	if (verbose):
		acc = hist.history["acc"][-1]
		loss = hist.history["loss"][-1]
		print("ACCURACY\t%.3f" % (acc))
		print("LOSS\t\t%.3f" % (loss))
	if (query_yes_no("Save weights?", default="yes")):
		model.save_weights('./data/weights/'+args.tmodel+'_weights.h5')
	return hist

def process_fmap(out, im, layer="", sz=sz, normalize=False):
	layer = "_"+layer
	out = np.resize(out, (sz, sz, 3))
	if (normalize):
		## Normalization
		out = (out-np.mean(out))/np.std(out)
	plt.subplot('121')
	plt.imshow(out)
	plt.axis('off')
	plt.xlabel("Feature map for layer " + layer[1:])
	values = list(map(lambda x : str(round(x, 1)), list(map(lambda f : f(out), [np.mean, np.std, np.median]))))
	plt.title("Mean = " + values[0] + " STD = " + values[1] + " Median = " + values[2])
	plt.subplot('122')
	plt.imshow(np.resize(im, (sz, sz, 3)))
	plt.axis('off')
	plt.xlabel("Input image")
	plt.show()

def save_fmap(out, layer="", sz=sz, normalize=False):
	layer = "_"+layer
	out = np.resize(out, (sz, sz, 3))
	if (normalize):
		## Normalization
		out = (out-np.mean(out))/np.std(out)
	plt.imshow(out)
	plt.axis('off')
	plt.title("Feature map for layer " + layer[1:])
	if (query_yes_no("Save feature map?", default="yes")):
		plt.savefig("Figures/"+args.tmodel+"/"+args.tmodel+"_feature_map_layer" + layer + ".png", bbox_inches="tight")

## Generator for training data
datagen_train = ImageDataGenerator(
	rescale=1.,
	featurewise_center=False,
	featurewise_std_normalization=False,
	## Normalization
	preprocessing_function=preprocess_image,
	## Data augmentation
	rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.2,
	horizontal_flip=True,
	data_format="channels_last",
)

## Generator for testing data
datagen_test = ImageDataGenerator(
	rescale=1.,
	featurewise_center=False,
	featurewise_std_normalization=False,
	## Normalization
	data_format="channels_last",
	preprocessing_function=preprocess_image,
)

if (args.trun == "training"):
	hist = run_nn(datagen_train, X_train, Y_train_c, Y_train, args.batch, training=True, verbose=True)
	labels = run_nn(datagen_train, X_val, Y_val_c, Y_val, args.batch, training=False, verbose=True)
if (args.trun == "testing"):
	k = min(1000, np.shape(X_test)[0])
	X_test = X_test[:k, :, :, :]
	Y_test_c = Y_test_c[:k, :]
	Y_test = Y_test[:k]
	labels = run_nn(datagen_test, X_test, Y_test_c, Y_test, args.batch, training=False, verbose=True)
if (args.trun == "deconv"):
	im_nb = 0
	layer_nb = layers.index(args.tlayer)
	print("** Layer: " + args.tlayer + " **")
	im = preprocess_image(np.expand_dims(X_test[im_nb, :, :, :], axis=0))
	out = model.predict([im])
	if (args.verbose == 1):
		print("#outputs = " + str(len(out)) + " of sizes:")
		print(list(map(np.shape, out)))
	## Feature images
	out = deconv_model.predict(out)
	if (args.verbose):
		process_fmap(out, im, layer=args.tlayer, normalize=(args.tmodel == "vgg"))
	save_fmap(out, layer=args.tlayer)
	## Weights
	weight = model.layers[layer_nb].get_weights()
	if (len(weight) > 0):
		for i in range(len(weight)):
			weight = weight[i]*255.
			if (args.verbose):
				process_fmap(weight, im, layer="weight" + str(i) + "_"+args.tlayer, normalize=(args.tmodel == "vgg"))
			save_fmap(weight, layer="weight" + str(i) + "_"+args.tlayer)
	else:
		print("Layer " + args.tlayer + " has no weights!")
	## Max activation
