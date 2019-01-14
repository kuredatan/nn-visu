#coding: utf-8

from __future__ import print_function
from keras import backend as K
import numpy as np
import matplotlib.pylab as plt
import cPickle as pickle
import sys
import os
from tqdm import tqdm

# SOURCE: Adapted from https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
# Start from noisy random image and get ONE image which maximizes the average activation
def grad_ascent(im, model, index, layer_name="", batch_size=32, niter=200, step=0.001):
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	if (layer_name):
	## Maximise a filter activation
		layer_output = layer_dict[layer_name].output
		try:
			loss = K.mean(layer_output[::, index])
		except:
			loss = K.mean(layer_output[index])
	else:
	## Maximise a specific class
		try:
			loss = K.mean(model.output[:, index])
		except:
			loss = K.mean(model.output[0][:, index])
	grads = K.gradients(loss, model.input)[0]
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
	iterate = K.function([model.input], [loss, grads])
	## Start from a gray noisy image
	im_ = np.random.random(np.shape(im)) * 20 + 128.
	## Gradient ascent
	for i in tqdm(range(niter)):
		loss_value, grads_value = iterate([im_])
		im_ += grads_value * step
	print("* Final loss: " + str(loss_value))
	return im_
