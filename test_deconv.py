## https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DeconvNet
#coding:utf-8
from __future__ import print_function
import sys
sys.path += ['./layers/', './utils/', './models/']
import numpy as np
import KerasDeconv
import cPickle as pickle
from utils import get_deconv_images
from utils import plot_deconv
from utils import plot_max_activation
from utils import find_top9_mean_act
import glob
import models
import cv2
import os

if __name__ == "__main__":

    ######################
    # Misc
    ######################
    model = None  # Initialise VGG model to None
    Dec = None  # Initialise DeconvNet model to None
    if not os.path.exists("./Figures/"):
        os.makedirs("./Figures/")

    ############
    # Load data
    ############
    sz = 32
    list_img = glob.glob("./data/Img/*.jpg*")
    assert len(list_img) > 0, "Put some images in the ./data/Img folder"
    if len(list_img) < 32:
        list_img = (int(32 / len(list_img)) + 2) * list_img
        list_img = list_img[:32]
    data = []
    for im_name in list_img:
        im = cv2.resize(cv2.imread(im_name), (sz, sz)).astype(np.float32)
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        im = im.transpose((2, 0, 1))
        data.append(im)
    data = np.array(data)

    ###############################################
    # Action 1) Get max activation for a secp ~/deconv_specificlection of feat maps
    ###############################################
    get_max_act = True
    if not model:
        model = models.Vonc(pretrained=True, deconv=True, sz=sz) 
        #model = load_model('./data/weights/vgg16_weights.h5')
        #model.compile(optimizer="sgd", loss='categorical_crossentropy')
    if not Dec:
        Dec = KerasDeconv.DeconvNet(model)
    if get_max_act:
        layers = [layer.name for layer in model.layers]
	layer = layers[4]
        d_act_path = './data/dict_top9_mean_act.pickle'
        d_act = {layer: {},
                 }
        ## Filters to check
        for feat_map in [3, 4]:
            d_act[layer][feat_map] = find_top9_mean_act(
                data, Dec, layer, feat_map, batch_size=32)
            d_act[layer][feat_map] = find_top9_mean_act(
                data, Dec, layer, feat_map, batch_size=32)
            with open(d_act_path, 'w') as f:
                pickle.dump(d_act, f)

    ###############################################
    # Action 2) Get deconv images of images that maximally activate
    # the feat maps selected in the step above
    ###############################################
    deconv_img = True
    if deconv_img:
        d_act_path = './data/dict_top9_mean_act.pickle'
        d_deconv_path = './data/dict_top9_deconv.pickle'
        get_deconv_images(d_act_path, d_deconv_path, data, Dec)

    raise ValueError

    ###############################################
    # Action 3) Get deconv images of images that maximally activate
    # the feat maps selected in the step above
    ###############################################
    plot_deconv_img = True
    if plot_deconv_img:
        d_act_path = './data/dict_top9_mean_act.pickle'
        d_deconv_path = './data/dict_top9_deconv.npz'
        target_layer = "convolution2d_10"
        plot_max_activation(d_act_path, d_deconv_path,
                            data, target_layer, save=True)

    ###############################################
    # Action 4) Get deconv images of some images for some
    # feat map
    ###############################################
    deconv_specific = False
    img_choice = False  # for debugging purposes
    if deconv_specific:
        if not model:
            model = load_model('./data/vgg16_weights.h5')
        if not Dec:
            Dec = KerasDeconv.DeconvNet(model)
        target_layer = "convolution2d_13"
        feat_map = 12
        num_img = 25
        if img_choice:
            img_index = []
            assert(len(img_index) == num_img)
        else:
            img_index = np.random.choice(data.shape[0], num_img, replace=False)
        plot_deconv(img_index, data, Dec, target_layer, feat_map)
