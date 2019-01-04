#coding:utf-8

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torch
import cv2
from imagenet1000 import imagenet1000

## CREDIT: https://blog.plon.io/tutorials/cifar-10-classification-using-keras-tutorial/
# functions to show 5*nrows random images from set X with labels Y (CIFAR-10: 10 classes)
def print_images(X, Y, num_classes=10, nrows=2):
	fig = plt.figure(figsize=(10,3))
	## For CIFAR-10
	if (num_classes == 10):
		class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	## For ImageNet
	elif (num_classes == 1000):
		class_names = imagenet1000.values()
	tick = 0
	for i in range(num_classes):
		idx = np.where(Y[:]==i)[0]
		if (len(idx) == 0):
			continue
		ax = fig.add_subplot(nrows, 5, 1 + tick, xticks=[], yticks=[])
		tick += 1
		features_idx = X[idx,:]
		img_num = np.random.randint(features_idx.shape[0])
		im = features_idx[img_num,::]
		ax.set_title(class_names[i])
		plt.imshow(im)
	plt.show()

# Code adapted from https://github.com/pedrodiamel/nettutorial/blob/master/pytorch/pytorch_visualization.ipynb
## Visualize feature maps/kernels from layer
def vistensor(layer, layer_name, ch=0, allkernels=False, nrow=20, padding=1):
	layer = torch.from_numpy(layer).float()
	n,c,w,h = layer.shape
	if allkernels: layer = layer.view(n*c,-1,w,h)
	elif c != 3: layer = layer[:,ch,:,:].unsqueeze(dim=1) 
	rows = np.min( (layer.shape[0]//nrow + 1, 64 )  )    
	grid = make_grid(layer, nrow=nrow, normalize=True, padding=padding)
	plt.figure( figsize=(nrow,rows) )
	plt.title("\nVisualization of tensor \'" + layer_name + "\'")
	plt.imshow(grid.numpy().transpose((1, 2, 0)))
	plt.show()

def plot_kernels(model, layer_name):
	layer = model.get_layer(layer_name).get_weights()
	if (len(layer) == 0):
		print("Layer \'" + layer_name + "\' has no weights! See model summary below")
		print(model.summary())
		raise ValueError
	vistensor(layer[0], layer_name)

def normalize_input(im, sz=224):
	if (str(type(im)) == "<type \'str\'>"):
		im = cv2.imread(im)
	im = cv2.resize(im, (sz, sz)).astype(np.float32)
	## Values from VGG authors
	#im[:,:,0] -= 103.939
	#im[:,:,1] -= 116.779
	#im[:,:,2] -= 123.68
	#im = (im-np.mean(im))*255./np.std(im)
	#im = np.expand_dims(im, axis=0)
	return im
