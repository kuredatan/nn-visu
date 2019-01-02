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
from imagenet1000 import imagenet1000
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
                    help='Import initialized weights: in {0, 1}.')
parser.add_argument('--lr', type=float, default=0.1, metavar='L',
                    help='Learning rate in (0,1).')
parser.add_argument('--decay', type=float, default=1e-6, metavar='C',
                    help='Decay rate in (0,1).')
parser.add_argument('--momentum', type=float, default=0.9, metavar='U',
                    help='Momentum.')
parser.add_argument('--layer', type=str, default="conv1-1", metavar='Y',
                    help='Name of the layer to deconvolve.')
parser.add_argument('--verbose', type=int, default=0, metavar='V',
                    help='Whether to print things or not: in {0, 1}.')
parser.add_argument('--subtask', type=str, default="", metavar='S',
                    help='Sub-task for deconvolution.')
parser.add_argument('--tdeconv', type=str, default="keras", metavar='K',
                    help='Choice of implementation for deconvolution: Mihai Dusmanu\'s (\'custom\') or DeepLearningImplementations (\'keras\').')
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

if (args.tmodel == "vgg"):
	sz = 32#224
else:
	sz = 32

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
	labels = [283, 281, 285, 281, 284, 282, 282, 281, 281, 281, 282]
	if len(list_img) < args.batch:
		list_img = (int(args.batch / len(list_img)) + 2) * list_img
		list_img = list_img[:args.batch]
		labels = (int(args.batch / len(labels)) + 2) * labels
		labels = labels[:args.batch]
	data = np.array([normalize_input(im_name, sz) for im_name in list_img])
	if (len(np.shape(data)) > 4):
		X_test = data.reshape((np.shape(data)[0], np.shape(data)[2], np.shape(data)[3], np.shape(data)[4]))
	else:
		X_test = data
	## Source for ImageNet labels: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
	Y_test = np.array(labels)

## Decomment to print 10 random images from X_train
#print_images(X_train, Y_train, num_classes=10, nrows=2)

if (args.trun != "training" and args.tdata == "CATS"):
	num_classes = 1000#300
else:
	num_classes = 10#1000

## Preprocessing
## CREDIT: https://keras.io/preprocessing/image/
X_test = X_test.astype('float32')
X_train = X_train.astype('float32')
X_train /= 255.0
X_test /= 255.0
Y_train_c = to_categorical(Y_train, num_classes)
Y_test_c = to_categorical(Y_test, num_classes)
Y_test_c = Y_test.astype('float32')
Y_train_c = Y_train.astype('float32')

## Cut X_train into training and validation datasets
p = 0.30
n = np.shape(X_train)[0]
in_val = sample(range(n), int(p*n))
in_train = list(set(in_val).symmetric_difference(range(n)))
X_val = X_train[in_val, :, :, :]
Y_val_c = Y_train_c[in_val, :]
X_train = X_train[in_train, :, :, :]
Y_train_c = Y_train_c[in_train, :]

d_models = {"conv": models.Conv, "vgg": models.VGG_16, "conv2": models.Conv2}
d_dmodels = {"conv": deconv_models.Conv, "vgg": deconv_models.VGG_16, "conv2": deconv_models.Conv2}

## NN model
model = d_models[args.tmodel](pretrained=(args.trun=="testing" and args.trained), deconv=args.trun == "deconv")
if (args.trun != "deconv"):
	model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

## "Deconvoluted" version of NN models
if (args.tdeconv == "custom" and args.trun == "deconv"):
	deconv_model = d_dmodels[args.tmodel](pretrained=(args.trun=="deconv" and args.trained), layer=args.layer if (args.trun=="deconv") else None)
	deconv_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
if (args.tdeconv == "keras" and args.trun == "deconv"):
	## Or the implementation of DeconvNet in Keras
	deconv_model = DeconvNet(model)

## Print kernels in a given layer (for instance "conv1-1")
layers = [layer.name for layer in model.layers]
if (args.verbose == 1):
	print("Layer names for model " + args.tmodel + ":\n")
	print(layers)
	print("______________________\nSummary:\n\n")
	print(model.summary())
#plot_kernels(model, layers[0])

###########################################
## TRAINING/TESTING/DECONV PIPELINES     ##
###########################################

if (args.trun == "training"):
	datagen_train = ImageDataGenerator(
		## NOT to use https://github.com/keras-team/keras/issues/3477
		#rescale=1. / 255,
		featurewise_center=True,
		featurewise_std_normalization=True,
		## Normalization
		preprocessing_function=lambda x : normalize_input(x, sz),
		## Data augmentation
		rotation_range=20,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=True,
		data_format="channels_last",
	)
	datagen_val = deepcopy(datagen_train)
	# compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied)
	datagen_train.fit(X_train)
	datagen_val.fit(X_val)
	generator = datagen_train.flow(X_train, Y_train_c, batch_size=args.batch)
	# fits the model on batches with real-time data augmentation:
	hist = model.fit_generator(datagen_train.flow(X_train, Y_train_c, batch_size=args.batch),
		shuffle=True,
		steps_per_epoch=int((1-p)*n)//args.batch,
		epochs=args.epoch,
		validation_data=datagen_val.flow(X_val, Y_val_c),
		validation_steps=int(p*n)//args.batch,
		#data_format="channels_last",
		#callbacks=[EarlyStopping(min_delta=0.001, patience=10)],
		verbose=2)
	print("* Final Accuracy: %.3f" % hist.history["acc"][-1])
	print("* Final (Validation) Accuracy: %.3f" % hist.history["val_acc"][-1])
	print("* Final Loss: %.3f" % hist.history["loss"][-1])
	print("* Final (Validation) Loss: %.3f" %hist.history["val_loss"][-1])
	model.save_weights('./data/weights/'+args.tmodel+'_weights.h5')
if (args.trun == "testing"):
	datagen_test = ImageDataGenerator(
		## NOT to use https://github.com/keras-team/keras/issues/3477
		#rescale=1. / 255,
		featurewise_center=True,
		featurewise_std_normalization=True,
		## Normalization
		data_format="channels_last",
		preprocessing_function=lambda x : normalize_input(x, sz),
	)
	datagen_test.fit(X_test)
	scores = model.evaluate_generator(datagen_test.flow(X_test, Y_test_c, batch_size=args.batch),
		steps=np.shape(X_test)[0]//args.batch,verbose=2)
	labels = model.predict_generator(datagen_test.flow(X_test, Y_test_c, batch_size=args.batch),
		steps=np.shape(X_test)[0]//args.batch,verbose=2)
	if (args.verbose):
		print(model.summary())
	print("* Predicted Labels:")
	print([np.argmax(labels[i]) for i in range(np.shape(labels)[0])])
	print("* Real Labels:")
	print(Y_test.T)
	print('* Test loss: %.2f' % scores[0])
	print('* Test accuracy: %.2f' % scores[1])
if (args.trun == "deconv"):
	out = model.predict(X_test)
	k = min(np.shape(out[0])[0], 5)
	labels = [imagenet1000[np.argmax(out[0][i])] for i in range(k)]
	print("* First "+str(k)+" Predicted Classes = \n" + str(labels))
	print("* Associated Real Classes = \n" + str([imagenet1000[i] for i in Y_test[:k].T[0].tolist()]))
	print(np.shape(out[0]),np.shape(out[1]),np.shape(out[2]))
	if (args.tdeconv == "keras"):
		out = deconv_model.get_deconv(out[0], layers[-1])#target_layer)
	else:
		out = deconv_model.predict(out)
	plt.figure(figsize=(20, 20))
	plt.imshow(out)
	plt.show()
	## Save output feature map
	#If you want to reconstruct from a single feature map / activation, you can
	# simply set all the others to 0. (in file "models/deconv_models.py")
	plt.savefig(folder + "fmap_" + str(i) + ".png", bbox_inches="tight")
	if (args.subtask == "max_activation"):
		get_max_act = True
		if get_max_act:
			if not model:
				model = load_model('./Data/vgg16_weights.h5')
			if not Dec:
				Dec = KerasDeconv.DeconvNet(model)
		d_act_path = './Data/dict_top9_mean_act.pickle'
		d_act = {"convolution2d_13": {},
			 "convolution2d_10": {}
			 }
		for feat_map in range(10):
			d_act["convolution2d_13"][feat_map] = find_top9_mean_act(
				data, Dec, "convolution2d_13", feat_map, batch_size=32)
			d_act["convolution2d_10"][feat_map] = find_top9_mean_act(
				data, Dec, "convolution2d_10", feat_map, batch_size=32)
			with open(d_act_path, 'w') as f:
				pickle.dump(d_act, f)


###################################################################
###################################################################

    ###############################################
    # Action 1) Get max activation for a secp ~/deconv_specificlection of feat maps
    ###############################################


    ###############################################
    # Action 2) Get deconv images of images that maximally activate
    # the feat maps selected in the step above
    ###############################################
#    deconv_img = True
#    if deconv_img:
#        d_act_path = './Data/dict_top9_mean_act.pickle'
#        d_deconv_path = './Data/dict_top9_deconv.pickle'
#        if not model:
#            model = load_model('./Data/vgg16_weights.h5')
#        if not Dec:
#            Dec = KerasDeconv.DeconvNet(model)
#        get_deconv_images(d_act_path, d_deconv_path, data, Dec)

    ###############################################
    # Action 3) Get deconv images of images that maximally activate
    # the feat maps selected in the step above
    ###############################################
#    plot_deconv_img = True
#    if plot_deconv_img:
#        d_act_path = './Data/dict_top9_mean_act.pickle'
#        d_deconv_path = './Data/dict_top9_deconv.npz'
#        target_layer = "convolution2d_10"
#        plot_max_activation(d_act_path, d_deconv_path,
#                            data, target_layer, save=True)
#
    ###############################################
    # Action 4) Get deconv images of some images for some
    # feat map
    ###############################################
#    deconv_specific = False
#    img_choice = False  # for debugging purposes
#    if deconv_specific:
#        if not model:
#            model = load_model('./Data/vgg16_weights.h5')
#        if not Dec:
#            Dec = KerasDeconv.DeconvNet(model)
#        target_layer = "convolution2d_13"
#        feat_map = 12
#        num_img = 25
#        if img_choice:
#            img_index = []
#            assert(len(img_index) == num_img)
#        else:
#            img_index = np.random.choice(data.shape[0], num_img, replace=False)
#        plot_deconv(img_index, data, Dec, target_layer, feat_map)
