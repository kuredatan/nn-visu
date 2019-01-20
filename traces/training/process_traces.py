#coding: utf-8

import numpy as np
import os
import matplotlib.pyplot as plt

if (not os.path.exists("mat_vonc.csv")):
	with open("./vonc.txt") as f:
		lines = f.readlines()
		nepochs=len(lines)
		mat = np.zeros((nepochs, 4))
		i = 0
		header = "loss,acc,val_loss,val_acc"
		for x in lines:
			els = list(filter(lambda y : y != "-", x.split(" ")))
			idx = els.index("loss:")
			mat[i, :] = [els[idx+1], els[idx+3], els[idx+5], els[idx+7]]
			i += 1
		np.savetxt(X=mat, fname="mat_vonc.csv", delimiter=",", header=header)
else:
	mat = np.loadtxt("mat_vonc.csv", delimiter=",")
	nepochs = np.shape(mat)[0]
	epochs = range(1, nepochs+1)
	acc = mat[:, 1]
	val_acc = mat[:, 3]
	loss = mat[:, 0]
	val_loss = mat[:, 2]
	plt.subplot('121')
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy for model Vonc')
	plt.legend()
	plt.subplot('122')
	plt.plot(epochs, loss, 'ro', label='Training loss')
	plt.plot(epochs, val_loss, 'r', label='Validation loss')
	plt.title('Training and validation loss for model Vonc')
	plt.legend()
	plt.savefig("training_acc_vonc.png", bbox_inches="tight")
	plt.show()
	
