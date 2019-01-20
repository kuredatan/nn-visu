#coding: utf-8

from __future__ import print_function
from keras import backend as K
import numpy as np
import matplotlib.pylab as plt
import cPickle as pickle
import sys
import os
from tqdm import tqdm

## SOURCE: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
## Tiled version for improved results
def calc_grad_tiled(img, iterate, tile_size=512):
    sz = tile_size
    img = img[0,:,:,:]
    h, w = np.shape(img)[1:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            _, g = iterate([sub])
            grad[y:y+sz,x:x+sz] = g
    g = np.roll(np.roll(grad, -sx, 1), -sy, 0)
    g = np.expand_dims(img, axis=0)
    return g

## SOURCE: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb?fbclid=IwAR231VV8fMHov8-NoDz8dwUiEVkyXWIdqIdN1VDvWWBJj1usvHWEWkzKb7o
## Normalize image for visualization
def plot_grad_ascent(img):
	#img = (img-np.mean(img))/max(np.std(img), 1e-4)*0.1 + 0.5
	img /= np.max(img)
	#img = np.asarray(img, dtype="uint8")#np.asarray(np.clip(img, 0, 1), dtype="uint8")
	print(type(img))
	print(type(img.resize(np.shape(img)[1:])))
	img = np.resize(img, np.shape(img)[1:])
	plt.imshow(img)
	plt.savefig("grad_ascent_plot.png", bbox_inches="tight")
	plt.show()

# SOURCE: Adapted from https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
# Start from noisy random image and get ONE image which maximizes the average activation
## SOURCE: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
## octave_n = 3
def grad_ascent(im, model, index, layer_name="", batch_size=32, niter=20, step=1.0, octave_n=1, octave_scale=1.4):
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
	#grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
	iterate = K.function([model.input], [loss, grads])
	## Start from a gray noisy image
	## SOURCE of this line: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
	im_ = np.random.uniform(size=np.shape(im)) + 100.0
	#im_ = np.random.random(np.shape(im)) * 20 + 128.
	## Gradient ascent
	## SOURCE of the tiled version: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
	for octave in range(octave_n):
		if (octave>0):
			hw = np.float32(np.shape(img)[1:2])*octave_scale
			img = np.resize(img, hw)
		for i in tqdm(range(niter)):
			if (octave_n > 1):
				grads_value = calc_grad_tiled(im_, iterate)
				loss_value = -1
			else:
				loss_value, grads_value = iterate([im_])
			## Normalizing
			grads_value /= np.std(grads_value) + 1e-8
			im_ += grads_value * step
	print("* Final optimized value: " + str(loss_value))
	#plot_grad_ascent(im_)
	return im_
