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
import keras.backend as K
import models
import deconv_models
import argparse
import glob
import cv2
import os
from imagenet1000 import imagenet1000
from cats import dict_labels_cats
from utils import grad_ascent
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import random

## Training models
# python2.7 process_model.py --tmodel vonc --tdata CIFAR-10 --trun training --trained 0 --epoch 250 --lr 0.0001 --optimizer Adam --batch 64
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
                    help='In ["conv2", "conv", "vonc", "vgg16", "resnet50"] ONLY.')
parser.add_argument('--trun', type=str, default='training', metavar='R',
                    help='In ["training", "testing", "deconv"].')
parser.add_argument('--tdata', type=str, default='CIFAR-10', metavar='D',
                    help='In ["CIFAR-10", "CATS", "siamese"].')
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
parser.add_argument('--subtask', type=str, default="fixed", metavar='S',
                    help='Sub-task for deconvolution: fixed layer (use the layer specified in --tlayer option) or compare layers (all convolutional and pooling layers).')
parser.add_argument('--step', type=float, default=0.0001, metavar='P',
                    help='Step for gradient ascent when visualizing feature maps/layers.')
parser.add_argument('--loss', type=str, default="categorical_crossentropy", metavar='O',
                    help='Choice of loss function, among those supported by Keras.')
parser.add_argument('--nb', type=str, default="", metavar='B',
                    help='Number of experiment in final pipeline (rename automatically the resulting figures).')
parser.add_argument('--all', type=int, default=0, metavar='A',
                    help='Automatically saving all figures and code ["0" or "1"].')
args = parser.parse_args()

folder = "./traces/"
if not os.path.exists(folder):
	os.mkdir(folder)
if not os.path.exists(folder+"testing/"):
        os.makedirs(folder+"testing/")
if not os.path.exists(folder+"training/"):
        os.makedirs(folder+"training/")

if not os.path.exists("./Figures/"):
        os.makedirs("./Figures/")
if not os.path.exists("./Figures/"+args.tmodel+"/"):
        os.makedirs("./Figures/"+args.tmodel+"/")

#if not os.path.exists("./Figures/exp/exp_"+args.tmodel+"/reconst"):
#        os.makedirs("./Figures/exp/exp_"+args.tmodel+"/reconst")
#if not os.path.exists("./Figures/exp/exp_"+args.tmodel+"/outputs"):
#        os.makedirs("./Figures/exp/exp_"+args.tmodel+"/outputs")

folder = "./slides+report/"
if not os.path.exists(folder):
        os.makedirs(folder)
if not os.path.exists(folder +"contributions/"):
        os.makedirs(folder +"contributions/")
folder += "figures_/"
if not os.path.exists(folder):
        os.makedirs(folder)
folder += "bow_analysis/"
if not os.path.exists(folder):
        os.makedirs(folder)

folder = "./data/bow_sift_comp/"
folders = [folder+'bow/', folder+'corresp/', folder+'harris/']
for f in folders:
	if not os.path.exists(f):
		os.makedirs(f)

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

pred_argmax = lambda y : [imagenet1000[np.argmax(y[i, :])] for i in range(len(y))]

import unicodedata

if (args.tmodel == "vgg16" or args.tmodel == "resnet50"):
	sz = 224
	if (args.tmodel == "vgg16"):
		from keras.applications.vgg16 import preprocess_input
		from keras.applications.vgg16 import decode_predictions
	if (args.tmodel == "resnet50"):
		from keras.applications.resnet50 import preprocess_input
		from keras.applications.resnet50 import decode_predictions
	## Gets the highest probability class label
	decode_predict = lambda y : [unicodedata.normalize('NFKD', decode_predictions(np.resize(yy, (1, num_classes)))[0][0][1]).encode('ascii','ignore') for yy in y]
	preprocess_image = lambda x : resize(x, (np.shape(x)[0], sz, sz, 3)) if (len(np.shape(x)) == 4) else resize(x, (1, sz, sz, 3))
	## preprocess_input(resize(x, (np.shape(x)[0], sz, sz, 3)))
else:
	sz = 32
	training_means = [np.mean(X_train[:,:,i].astype('float32')) for i in range(3)]
	preprocess_image = lambda x : normalize_input(x, sz, training_means)
	decode_predict = pred_argmax

## CREDIT: Keras training on CIFAR-10 
## https://gist.github.com/giuseppebonaccorso/e77e505fc7b61983f7b42dc1250f31c8

## CREDIT: https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DeconvNet
if (args.trun != "training" and args.tdata != "CIFAR-10"):
	if (args.tdata == "CATS"):
		list_img = glob.glob("./data/cats/*.jpg*")
		assert len(list_img) > 0, "Put some images in the ./data/cats folder"
		labels = [dict_labels_cats[img] for img in list_img]
	if (args.tdata == "siamese"):
		list_img = glob.glob("./data/siamese/*.jpg*")
		assert len(list_img) > 0, "Put some images in the ./data/siamese folder"
		labels = [284]*len(list_img)
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

d_models = {"conv": models.Conv, "vgg16": models.VGG_16, "conv2": models.Conv2, "vonc": models.Vonc, "resnet50": models.ResNet_50}
d_dmodels = {"conv": deconv_models.Conv, "vgg16": deconv_models.VGG_16, "conv2": deconv_models.Conv2, "vonc": deconv_models.Vonc, "resnet50": deconv_models.ResNet_50}

## NN model
model = d_models[args.tmodel](pretrained=args.trained>0, deconv=args.trun in ["deconv", "final"], sz=sz, layer=args.tlayer)
if (not args.trun in ["deconv", "final"]):
	model.compile(loss=args.loss, optimizer=optimizer, metrics=['accuracy'])

## Print kernels in a given layer
layers = [layer.name for layer in model.layers]
if (args.verbose == 1):
	print("Layer names for model " + args.tmodel + ":")
	print(layers)
	print("______________________\nSummary:")
	print(model.summary())

#layer = layers[1]
#print("Plotting kernel from layer \'" + layer + "\'")
#plot_kernels(model, layer)

## "Deconvoluted" version of NN models
if (args.trun == "deconv" or args.trun == "final"):
	if (not args.tlayer in layers and args.trun == "deconv"):
		print(args.tlayer + " is not in layer list: " + str(layers))
		raise ValueError
	deconv_model = d_dmodels[args.tmodel](pretrained=args.trained>0, layer=args.tlayer, sz=sz)
	deconv_model.compile(loss=args.loss, optimizer=optimizer, metrics=['accuracy'])

###########################################
## TRAINING/TESTING/DECONV PIPELINES     ##
###########################################

def run_nn(datagen, X, Y_c, Y, batch_size, X_val=None, Y_val_c=None, training=False, kmin=10):
	datagen.fit(X)
	labels = []
	n = np.shape(X)[0]
	epochs = args.epoch if (training) else 1
	Y_test = []
	batch = 0
	if (training):
		hist = model.fit_generator(datagen.flow(X, Y_c, batch_size=batch_size), verbose=1,
			epochs=epochs,shuffle=True,steps_per_epoch=np.shape(X)[0]//batch_size,
			validation_data=datagen.flow(X_val, Y_val_c, batch_size=batch_size),
			validation_steps=np.shape(X_val)[0]//batch_size,
			callbacks=[EarlyStopping(monitor="loss", min_delta=0.001, patience=3)])
	else:
		for x_batch, y_batch in datagen.flow(X, Y_c, batch_size=batch_size):
			if (args.verbose):
				print("Batch #" + str(batch+1) + "/" + str(n/batch_size+1))
			predictions = model.predict(x_batch, batch_size, verbose=1)
			mn = len(predictions)
			pred_ = decode_predict(predictions)
			labels += pred_
			Y_batch = pred_argmax(y_batch)
			Y_test += Y_batch
			#print("pred= ", pred_, type(pred_[0]))
			#print("Y_batch= ", Y_batch, type(Y_batch[0]))
			acc = np.array([int(pred_[i] == Y_batch[i]) for i in range(mn)])#np.array(pred_)==np.array(Y_batch)
			acc = np.sum(acc)/float(mn)
			#print(acc)
			#raise ValueError
			if (args.verbose):
				print(str(batch_size*(batch+1)) + "/" + str(n) + ": acc = " + str(acc))
			if batch >= n / batch_size:
				break
			else:
				batch += 1
	if (args.tdata == "CATS" and args.trun == "testing"):
		target_names = list(set([imagenet1000[l] for l in dict_labels_cats.values()]))
	else:
		target_names = imagenet1000.keys()
	print_conf_report = False
	if (print_conf_report):
		print('\n* * * Confusion Matrix')
		target = list(set(Y_test+labels))
		conf_mat = confusion_matrix(Y_test, labels, labels=target)
		print(conf_mat)
		print('\n* * * Classification Report')
		target_names = list(map(lambda x : imagenet1000[x], list(set(Y_test))))
		if (len(target_names) == 1):
			target_names += ["not "+target_names[0]]
		class_report = classification_report(Y_test, labels, target_names=target_names)
		print(class_report)
	names = ["model", "epochs", "batch", "optimizer", "lr"]
	feat = list(map(str, [args.tmodel, args.epoch, args.batch, args.optimizer, args.step]))
	caract = names[0]+"="+feat[0]
	for i in range(1, len(names)):
		caract += "_" + names[i]+"="+feat[i]
	if (print_conf_report):
		if (query_yes_no("Save confusion matrix/report?", all_=args.all, default="yes")):
			header = ""
			for t in target:
				header += t+","
			np.savetxt("./testing/conf_matrix_"+caract+".csv", conf_mat, header=header)
			## https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
			## read_dictionary = np.load('my_file.npy').item()
			np.save("./testing/report_"+caract+".npy", class_report) 
	if (not training):
		if (args.verbose == 1):
			acc = np.sum(np.array(labels) == np.array(Y_test))/float(len(labels))
			if (args.verbose):
				print(model.summary())
			k = min(np.shape(labels)[0], kmin)
			pred = [imagenet1000[label] for label in labels]
			real = [imagenet1000[y] for y in Y_test]
			print("")
			print("PREDICTED" + "\t"*3 + "REAL LABELS")
			for i in range(k):
				print(pred[i] + " - " + real[i])
			print('')
			print('* ACCURACY %.2f' % acc)
		return labels
	if (query_yes_no("Save weights?", default="yes", all_=args.all)):
		model.save_weights('./data/weights/'+args.tmodel+'_weights.h5')
	if (args.verbose == 1):
		acc = hist.history["acc"][-1]
		loss = hist.history["loss"][-1]
		print("ACCURACY\t%.3f" % (acc))
		print("LOSS\t\t%.3f" % (loss))
	## https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0
	## Loss curves
	acc = hist.history['acc']
	val_acc = hist.history['val_acc']
	loss = hist.history['loss']
	val_loss = hist.history['val_loss']
	epochs = range(1, len(acc) + 1)
	plt.subplot('121')
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy for model ' + args.tmodel)
	plt.legend()
	plt.subplot('122')
	plt.plot(epochs, loss, 'ro', label='Training loss')
	plt.plot(epochs, val_loss, 'r', label='Validation loss')
	plt.title('Training and validation loss for model ' + args.tmodel)
	plt.legend()
	mat = np.zeros((len(acc), 4))
	obj = [acc, val_acc, loss, val_loss]
	for i in range(4):
		mat[:, i] = obj[i]
	if (query_yes_no("Save loss/accuracy values?", default="yes", all_=args.all)):
		np.savetxt("loss_acc_"+caract+".csv", mat, header="acc,val_acc,loss,val_loss")
	fig = plt.gcf()	
	if (args.verbose):
		plt.show()
	if (query_yes_no("Save loss/accuracy curves?", default="yes", all_=args.all)):
		fig.savefig(args.tmodel+"_"+args.tdata+"_loss_acc_curves.png", bbox_inches="tight")
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
	fig = plt.gcf()
	if (query_yes_no("Save feature map?", default="yes", all_=args.all)):
		print("Saved in Figures/"+args.tmodel+"/"+args.tmodel+"_feature_map_layer" + layer + ".png")
		fig.savefig("Figures/"+args.tmodel+"/"+args.tmodel+"_feature_map_layer" + layer + ".png", bbox_inches="tight")

def save_inputs(filters, layer='', class_='', sz=sz, normalize=True, show=True, nb=args.nb):
	if (nb):
		nb = "_"+nb
	fig = plt.figure(figsize=(20,20))
	plt.axis('off')
	n = len(filters)
	for i in range(n):
		plt.subplot('1'+str(n)+str(i))
		plt.axis('off')
		x = np.resize(filters[i], (sz, sz, 3))
		if (normalize):
			# SOURCE: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
			x -= x.mean()
			x /= (x.std() + 1e-5)
			x *= 0.1
			x += 0.5
			x = np.clip(x, 0, 1)
			x *= 255
			x = np.clip(x, 0, 255).astype('uint8')
		plt.imshow(x)
	if (layer):
		plt.title("Reconstructed inputs for maximizing activation in layer " + layer)
	if (class_):
		plt.title("Reconstructed inputs for maximizing activation in class " + class_)
	fig = plt.gcf()
	if (show):
		plt.show()
	if (query_yes_no("Save feature map?", default="yes", all_=args.all)):
		fig.savefig("Figures/"+args.tmodel+"/"+args.tmodel+"_feature_map_layer" + layer + nb + ".png", bbox_inches="tight")
		print("Saved in Figures/"+args.tmodel+"/"+args.tmodel+"_feature_map_layer" + layer + nb + ".png")

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
	hist = run_nn(datagen_train, X_train, Y_train_c, Y_train, args.batch, X_val, Y_val_c, training=True)
if (args.trun == "testing"):
	k = min(1000, np.shape(X_test)[0])
	X_test = X_test[:k, :, :, :]
	Y_test_c = Y_test_c[:k, :]
	Y_test = Y_test[:k]
	labels = run_nn(datagen_test, X_test, Y_test_c, Y_test, args.batch, training=False)
if (args.trun == "deconv"):
	im_nb = 0
	layer_nb = layers.index(args.tlayer)
	print("** Layer: " + args.tlayer + " **")
	im = preprocess_image(np.expand_dims(X_test[im_nb, :, :, :], axis=0))
	out = model.predict([im])
	#save_fmap(out, layer="im="+str(im_nb)+"_output_model_" + args.tlayer)
	if (args.verbose == 1):
		print("#outputs = " + str(len(out)) + " of sizes:")
		print(list(map(np.shape, out)))
	## Feature images
	out = deconv_model.predict(out)
	if (args.verbose):
		process_fmap(out, im, layer=args.tlayer, normalize=(args.tmodel == "vgg16"))
	save_fmap(out, layer="im="+str(im_nb)+"_"+args.tlayer)
	if (False):
		## Weights
		weight = model.layers[layer_nb].get_weights()
		if (len(weight) > 0):
			for i in range(len(weight)):
				weight = weight[i]*255.
				if (args.verbose):
					process_fmap(weight, im, layer="weight" + str(i) + "_"+args.tlayer, normalize=(args.tmodel == "vgg16"))
				save_fmap(weight, layer="im="+str(im_nb)+"_weight" + str(i) + "_"+args.tlayer)
		else:
			print("Layer " + args.tlayer + " has no weights!")
		## Reconstruct image with highest average activation for filter of index filter_index
		## https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
		## Select n filters randomly and reconstruct inputs
		## <!> Fixed layer
		if (args.subtask == "fixed"):
			n = 1 # < 10
			m = 1
			rand_range = random.sample(range(m), n)
			filters = [grad_ascent(im, model, filter_index, layer_name=args.tlayer, batch_size=args.batch, step=args.step) for filter_index in rand_range]
		## <!> Compare different layers
		else:
			filter_index = 0
			layers = list(filter(lambda x : x[:6] != "interp" and x[:3] != "pos" and x[:5] != "input", map(lambda x : x.name, deconv_model.layers)))
			filters = [grad_ascent(sz, model, filter_index, layer_name=layer, batch_size=args.batch, step=args.step) for layer in layers]
		save_inputs(filters, layer="grad_ascent_im="+str(im_nb)+"_"+args.tlayer)
		## Reconstruct image with highest average activation for class
		## https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
		class_ = 284
		ntries = 1
		## Full model
		model = d_models[args.tmodel](pretrained=args.trained>0, sz=sz, deconv=True)
		imgs = [grad_ascent(sz, model, class_, batch_size=args.batch, step=args.step) for i in range(ntries)]
		save_inputs(imgs, layer="grad_ascent_im="+str(im_nb)+"_class="+str(class_), class_=imagenet1000[class_])
## Final pipeline: Call
## python2.7 process_model.py --tmodel conv --trained 1 --trun final --batch 32 --tdata siamese --lr 0.001 --optimizer Adam --loss categorical_crossentropy --epoch 10
## python2.7 process_model.py --tmodel vgg --trained 1 --trun final --batch 32 --tdata siamese --lr 0.001 --optimizer Adam --loss categorical_crossentropy --epoch 10 --all 1 --nb 302 --step 1 --tlayer block3_conv3
if (args.trun == "final"):
	print("* * Experiment #" + args.nb)
	## Parameters
	class_ = 284 # 'Siamese cat' class
	ntries = 1 #Nb of inputs to reconstruct
	n = 1 # < 10 Nb of features maps to show
	m = 3
	p = 0.30 ## proportion of validation set in training set
	todo = {"max_act_filter_btrain": False,
		"max_act_class_btrain": True,
		"train": 1,
		"max_act_filter_atrain": False,
		"max_act_class_atrain": True} ## Whether to perform each step
	## Show pictures if set to True
	show = False
	## STEP 2: Before training
	###### Get the reconstructed inputs yielding highest mean activation
	if (todo["max_act_filter_btrain"]):
		## A set of feature channels to visualize
		rand_range = random.sample(range(m), n)
		layers = list(filter(lambda x : x[:6] != "interp" and x[:3] != "pos" and x[:5] != "input" and x[:10] != "activation", map(lambda x : x.name, deconv_model.layers)))
		filters = [[grad_ascent(sz, model, filter_index, layer_name=layer, batch_size=args.batch, step=args.step) for filter_index in rand_range] for layer in layers]
		hf = ''
		for f in rand_range:
			hf += str(f)+','
		for i in range(len(filters)):
			save_inputs(filters[i], layer="b_training_layer="+layers[i]+"_filters="+hf, show=show)
	###### Get the reconstructed inputs yielding highest mean activation in 
	###### class_ output (wrt ImageNet labels) of softmax layer
	if (todo["max_act_class_btrain"]):
		## Full model
		model = d_models[args.tmodel](pretrained=args.trained>0, sz=sz, deconv=True, include_softmax = False)
		imgs = [grad_ascent(sz, model, class_, batch_size=args.batch, step=args.step) for i in range(ntries)]
		save_inputs(imgs, layer="b_training_class="+str(class_), class_=imagenet1000[class_], show=show)
	## STEP 3: Training with the considered class
	if (todo["train"]):
		Y_test_c = to_categorical(Y_test, num_classes)
		## Cut X_test into training and validation datasets
		n = np.shape(X_test)[0]
		in_val = sample(range(n), int(p*n))
		in_train = list(set(in_val).symmetric_difference(range(n)))
		X_val = X_test[in_val, :, :, :]
		Y_val = Y_test[in_val]
		Y_val_c = Y_test_c[in_val, :]
		X_test = X_test[in_train, :, :, :]
		Y_test = Y_test[in_train]
		Y_test_c = Y_test_c[in_train, :]
		if (args.tmodel != "vgg16" and args.tmodel != "resnet50"):
			model = d_models[args.tmodel](pretrained=args.trained>0, sz=sz, deconv=False)
		else:
			## Directly import the model from Keras
			if (args.tmodel == "vgg16"):
				from keras.applications.vgg16 import VGG16
				model = VGG16(include_top=True, weights="imagenet", classes=num_classes)
			if (args.tmodel == "resnet50"):
				from keras.applications.resnet50 import ResNet50
				model = ResNet50(include_top=True, weights="imagenet", classes=num_classes)
		model.compile(loss=args.loss, optimizer=optimizer, metrics=['accuracy'])
		## Remove data augmentation
		hist = model.fit(preprocess_image(X_test), Y_test_c, batch_size=args.batch, verbose=1,
				epochs=args.epoch,shuffle=True,
				validation_data=(preprocess_image(X_val), Y_val_c),
				callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.001, patience=3)])
		## SAVE WEIGHTS
		wname = './data/weights/'+args.tmodel+'_'+args.tdata+'_weights.h5'
		model.save_weights(wname)
		print("Saved weights at " + wname)
	## STEP 4
	if (todo["max_act_filter_atrain"]):
		###### Get the reconstructed inputs yielding highest mean activation
		filters = [[grad_ascent(sz, model, filter_index, layer_name=layer, batch_size=args.batch, step=args.step) for filter_index in rand_range] for layer in layers]
		for i in range(len(filters)):
			save_inputs(filters[i], layer="a_training_layer="+layers[i]+"_filters="+hf, show=show)
	###### Get the reconstructed inputs yielding highest mean activation in 
	###### class_ output (wrt ImageNet labels) of softmax layer
	if (todo["max_act_class_atrain"]):
		## Full model
		model = d_models[args.tmodel](pretrained=args.trained>0, sz=sz, weights_path=wname, deconv=True, include_softmax = False)
		imgs = [grad_ascent(sz, model, class_, batch_size=args.batch, step=args.step) for i in range(ntries)]
		save_inputs(imgs, layer="a_training_class="+str(class_), class_=imagenet1000[class_], show=show)
