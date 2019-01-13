#coding: utf-8
## SOURCE: https://stackoverflow.com/questions/44186042/keras-methods-to-enlarge-spartial-dimension-of-the-layer-output-blob

from keras import layers, models, utils
from keras.backend import tf as ktf

class Interp(layers.Layer):
    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        new_height, new_width = self.new_size
        resized = ktf.image.resize_images(inputs, [new_height, new_width],
                                          align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Interp, self).get_config()
        config['new_size'] = self.new_size
        return config


## Reshape layer
#try:
#	x = Lambda(lambda image: ktf.image.resize_images(image, (224, 224)))(inp)
#except :
#	# if you have older version of tensorflow
#	x = Lambda(lambda image: ktf.image.resize_images(image, 224, 224))(inp)

