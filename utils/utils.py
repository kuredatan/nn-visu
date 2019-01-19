#coding: utf-8

from __future__ import print_function
from keras import backend as K
import numpy as np
import matplotlib.pylab as plt
import cPickle as pickle
import sys
import os
from tqdm import tqdm

## SOURCE: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb?fbclid=IwAR231VV8fMHov8-NoDz8dwUiEVkyXWIdqIdN1VDvWWBJj1usvHWEWkzKb7o
## Normalize image for visualization
def plot_grad_ascent(img):
	img = (img-np.mean(img))/max(np.std(img), 1e-4)*0.1 + 0.5
	img = np.asarray(np.clip(img, 0, 1), dtype="uint8")
	img = np.resize(img, np.shape(img)[1:])
	plt.imshow(img)
	plt.savefig("grad_ascent_plot.png", bbox_inches="tight")
	plt.show()

# SOURCE: Adapted from https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
# Start from noisy random image and get ONE image which maximizes the average activation
def grad_ascent(im, model, index, layer_name="", batch_size=32, niter=20, step=1.0):
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
	## Normalizing the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
	iterate = K.function([model.input], [loss, grads])
	## Start from a gray noisy image
	im_ = np.random.random(np.shape(im)) * 20 + 128.
	## Gradient ascent
	for i in tqdm(range(niter)):
		loss_value, grads_value = iterate([im_])
		## Normalizing
		#grads_value /= (K.sqrt(K.mean(K.square(grads))) + 1e-8)
		im_ += grads_value * step
	print("* Final loss: " + str(loss_value))
	plot_grad_ascent(im_)
	return im_
