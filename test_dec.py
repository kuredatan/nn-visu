#coding:utf-8
import sys
sys.path += ['layers', 'utils', 'models']
from keras.models import Model
from keras.layers import Input, Conv2D
from deconv2D import Deconv2D
import models
import deconv_models
from pool_unpool import MaxPooling2D, UndoMaxPooling2D
from print_norm_utils import load_input, normalize_input
 
def forward_model():  
  inp = Input(shape = (sz, sz, 3))
  x = inp
 
  x = Conv2D(16, 3, padding = 'SAME', activation = 'relu', name = 'block1_conv1')(x)
  x, pos1 = MaxPooling2D(name = 'block1_pool')(x)
 
  x = Conv2D(32, 3, padding = 'SAME', activation = 'relu', name = 'block2_conv1')(x)
  x, pos2 = MaxPooling2D(name = 'block2_pool')(x)
 
  x = Conv2D(64, 3, padding = 'SAME', activation = 'relu', name = 'block3_conv1')(x)
 
  return Model(inputs = inp, outputs = [x, pos1, pos2])
 
# Now you can use forward_pass.predict([im]) to obtain the feature maps after block3_conv1
# and the pooling positions from block1_pool and block2_pool.
 
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

import numpy as np
import matplotlib.pyplot as plt
#sz = 224
sz = 32

if (False):
	backward_net = backward_model()
	backward_net.load_weights('my_weights.h5', by_name = True)
	forward_net = forward_model()
	forward_net.load_weights('my_weights.h5', by_name = True)

forward_net = models.Conv(pretrained=True, deconv=True)
backward_net = deconv_models.Conv(pretrained=True)

#im = load_input("./data/cats/cat1.jpg", sz)
im = normalize_input("./data/cats/cat1.jpg", sz)
out = forward_net.predict([im])
print(len(out))
print([np.shape(x) for x in out])
out = backward_net.predict(out)

## TODO
## Get activation zones
## Save feature maps
## Apply methods

out = np.resize(out, (sz, sz, 3))
print(np.shape(out), np.shape(im))
plt.subplot('121')
plt.imshow(out)
plt.subplot('122')
plt.imshow(np.resize(im, (sz, sz, 3)))
plt.show()
print(np.mean(out), np.std(out), np.median(out))
 
# Now you can use backward_pass.predict([fmaps, pos1, pos2]) to obtain the reconstructed
# input. If you want to reconstruct from a single feature map / activation, you can
# simply set all the others to 0.
