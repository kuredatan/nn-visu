#coding:utf-8
## Credit: Mihai Dusmanu

#python2.7 test.py
import sys
sys.path += ['./layers/', './utils/']

from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense
 
from deconv2D import Deconv2D
from pool_unpool import MaxPooling2D, UndoMaxPooling2D
from print_norm_utils import normalize_input
import numpy as np
from keras.optimizers import SGD
 
sz = 224
weights_path = None
 
def forward_model():  
	inp = Input(shape = (sz, sz, 3))
	x = inp

	conv1 = Conv2D(16, 3, padding = 'SAME', activation = 'relu', name = 'block1_conv1')(x)
	mp1, pos1 = MaxPooling2D(name = 'block1_pool')(conv1)
	x = Flatten()(conv1)
	conv1 = Dense(1000, activation='softmax')(x)

	conv2 = Conv2D(32, 3, padding = 'SAME', activation = 'relu', name = 'block2_conv1')(mp1)
	mp2, pos2 = MaxPooling2D(name = 'block2_pool')(conv2)
	x = Flatten()(conv2)
	conv2 = Dense(1000, activation='softmax')(x)

	conv3 = Conv2D(64, 3, padding = 'SAME', activation = 'relu', name = 'block3_conv1')(mp2)

	x = Flatten()(conv3)
	x = Dense(1000, activation='softmax')(x)
	return Model(inputs = inp, outputs = [x, conv3, conv2, conv1, pos1, pos2])

forward_net = forward_model()
if (weights_path):
	forward_net.load_weights(weights_path+'.h5', by_name = True)

# Now you can use forward_pass.predict([im]) to obtain the feature maps after block3_conv1
# and the pooling positions from block1_pool and block2_pool.
im_name = "./data/cats/cat1.jpg"
im = normalize_input(im_name, sz)
[x, conv3, conv2, conv1, pos1, pos2] = forward_net.predict([im])
print("Predicted class = " + str(np.argmax(x)))
 
# The Deconv2D layers should have the same name as the associated Conv2D layers.
# The shapes can be extracted from forward_net.summary().
def backward_model():
	inp = Input(batch_shape = (1, sz // 4, sz // 4, 64))
	x = inp

	x = Deconv2D(32, 3, padding = 'SAME', activation = 'relu', name = 'block3_conv1')(x)

	pos2 = Input(batch_shape = (1, sz // 4, sz // 4, 32))
	x = UndoMaxPooling2D((1, sz // 2, sz // 2, 32), name = 'block2_unpool')([x, pos2])
	x = Deconv2D(16, 3, padding = 'SAME', activation = 'relu', name = 'block2_conv1')(x)

	pos1 = Input(batch_shape = (1, sz // 2, sz // 2, 16))
	x = UndoMaxPooling2D((1, sz, sz, 16), name = 'block1_unpool')([x, pos1])
	x = Deconv2D(3, 3, padding = 'SAME', activation = 'relu', name = 'block1_conv1')(x)

	return Model(inputs = [inp, pos1, pos2], outputs = x)
 
backward_net = backward_model()
if (weights_path):
	backward_net.load_weights(weights_path+'.h5', by_name = True)
 
# Now you can use backward_pass.predict([fmaps, pos1, pos2]) to obtain the reconstructed
# input. If you want to reconstruct from a single feature map / activation, you can
# simply set all the others to 0.

out = backward_net.predict([x, pos1, pos2])
import matplotlib.pyplot as plt
plt.imshow(out)
plt.show()
