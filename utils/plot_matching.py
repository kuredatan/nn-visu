## SOURCE: Adapted from http://scikit-image.org/docs/dev/auto_examples/transform/plot_matching.html#sphx-glr-auto-examples-transform-plot-matching-py

import numpy as np
from matplotlib import pyplot as plt

from skimage import data
from skimage.util import img_as_float
from skimage.feature import (corner_harris, corner_subpix, corner_peaks, plot_matches)
from skimage.transform import warp, AffineTransform
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.measure import ransac

def rgb2grey(im):
	return np.float32(np.dot(im[...,:3], [0.299, 0.587, 0.114])/255.0)

# extract corners using Harris' corner measure
def extract_corners(im):
	im = rgb2grey(im)
	coords = corner_peaks(corner_harris(im), threshold_rel=0.001, min_distance=1)
	#coords = corner_subpix(im, coords, window_size=9)
	coords = coords[~np.isnan(coords).any(axis=1)]
	return coords

def gaussian_weights(window_sh, sigma=1):
    a, b = window_sh[:2]
    y, x = np.mgrid[-(a//2):(a//2)+1, -(b//2):(b//2)+1]
    g = np.zeros(y.shape, dtype=np.double)
    g[:] = np.exp(-0.5 * (x**2 / sigma**2 + y**2 / sigma**2))
    g /= 2 * np.pi * sigma * sigma
    return g

def match_corner(im1, im2, coord1, coord2, window_ext=5):
    r, c = np.round(coord1).astype(np.intp)
    window_o = im1[max(0,r-window_ext):min(np.shape(im1)[1], r+window_ext+1),
                   max(0,c-window_ext):min(np.shape(im1)[1], c+window_ext+1), :]

    # compute sum of squared differences to all corners in warped image
    SSDs = []
    for cr, cc in coord2:
        cr, cc = np.intp(cr), np.intp(cc)
        window_w = im2[max(0,cr-window_ext):min(np.shape(im2)[1], cr+window_ext+1),
                      max(cc-window_ext,0):min(np.shape(im2)[1], cc+window_ext+1), :]
        # weight pixels depending on distance to center pixel
        weights = gaussian_weights(np.shape(window_o), 3)
        weights = np.dstack((weights, weights, weights))
        sh = np.shape(weights)
        SSD = np.sum(weights * (np.resize(window_o, sh) - np.resize(window_w, sh))**2)
        SSDs.append(SSD)

    # use corner with minimum SSD as correspondence
    min_idx = np.argmin(SSDs)
    return coord2[min_idx]

def ransac_(src, dst):
	# robustly estimate affine transform model with RANSAC
	model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=3,
		                       residual_threshold=2, max_trials=100)
	return model_robust, inliers

# find correspondences using simple weighted sum of squared differences
def find_correspondences(im1, im2, coords1, coords2):
	src = []
	dst = []
	for coord in coords1:
	    src.append(coord)
	    dst.append(match_corner(im1, im2, coord, coords2))
	src = np.array(src)
	dst = np.array(dst)
	# robustly estimate affine transform model with RANSAC
	model_robust, inliers = ransac_(src, dst)
	return inliers, src, dst

## SOURCE: practicals #1 from Andrea Vedaldi and Andrew Zisserman, Gul Varol and Ignacio Rocco
def plot_correspondences(im1, im2, frames1, frames2, matches, fmap_name):
	# plot matches
	plt.imshow(np.concatenate((im1,im2),axis=1))
	for i in range(np.shape(matches)[0]):
		k, j=matches[i,:]
		# plot dots at feature positions
		plt.gca().scatter([frames1[k,0],im1.shape[1]+frames2[j,0]], [frames1[k,1],frames2[j,1]], s=5, c=np.array([[0,1,0]]))
		# plot lines
		plt.plot([frames1[k,0],im1.shape[1]+frames2[j,0]],[frames1[k,1],frames2[j,1]],linewidth=0.5)
	plt.axis('off')
	plt.title('RANSAC filtered correspondances with closest image')
	#plt.savefig(fmap_name+"_ransac.png", bbox_inches="tight")
	plt.savefig("ransac.png", bbox_inches="tight")
	plt.show()
