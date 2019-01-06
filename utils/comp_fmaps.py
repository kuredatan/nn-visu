#coding:utf-8

## Adapted from practicals from Andrea Vedaldi and Andrew Zisserman by Gul Varol and Ignacio Rocco (HMW1)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from sklearn.neighbors import KDTree, DistanceMetric
from sklearn.cluster import MiniBatchKMeans
import os
import glob
import cyvlfeat
from scipy.ndimage import convolve
from cyvlfeat.plot import plotframes
from PIL import Image
from scipy.misc import imresize

sz = 32

thres = 0.01

# change some default matplotlib parameters
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.dpi'] = 120

# ignore warnings
warnings.filterwarnings('ignore')

ff = "/home/reda/Projets20182019/Vision/Projet/interpretation/DeconvNet/"
folder = ff+"data/bow_sift_comp/"
folders = [folder+'bow/', folder+'corresp/', folder+'harris/']
for f in folders:
	if not os.path.exists(f):
		os.mkdir(f)

###########
## TOOLS ##
###########

def load_input(im_name):
	print(im_name)
	img = Image.open(im_name)
	img.load()
	im = np.asarray(img, dtype=np.float32)
	im = imresize(im, (sz, sz, 3))
	return im

def rgb2gray(rgb):
	im = np.float32(np.dot(rgb[...,:3], [0.299, 0.587, 0.114])/255.0)
	return im

def plot_bovw(hst, title="mystery image"):
	plt.hist(np.histogram(hst))
	plt.title("Bag-of-Visual-Words with " + title)
	plt.xlabel("Word ID")
	plt.ylabel("\'Count\'")
	plt.show()

def plot_bovw_compare(query_hist, hst, score):
	plt.figure(figsize=(10, 4.19))
	plt.subplot('121')
	plt.hist(np.histogram(query_hist))
	plt.title("Bag-of-Visual-Words with input image/feature map")
	plt.xlabel("Word ID")
	plt.ylabel("\'Count\'")
	plt.subplot('122')
	plt.hist(np.histogram(hst))
	plt.title("Bag-of-Visual-Words with closest image (score = "+ str(round(score, 2))+")")
	plt.xlabel("Word ID")
	plt.ylabel("\'Count\'")
	plt.show()

def plot_image_compare(query, im):
	if (np.max(query) > 1):
		query = query/255.
	if (np.max(im) > 1):
		im = im/255.
	plt.figure(figsize=(10, 4.19))
	plt.subplot('121')
	plt.imshow(query)
	plt.title("input image/feature map")
	plt.xlabel("")
	plt.ylabel("")
	plt.subplot('122')
	plt.imshow(im)
	plt.title("Closest image")
	plt.xlabel("")
	plt.ylabel("")
	plt.show()

def compute_SIFT(im):
	sift = cyvlfeat.sift.sift(rgb2gray(im),peak_thresh=thres,compute_descriptor=True)
	return sift

def get_descrs(im):
	descrs = np.asarray(compute_SIFT(im)[1], dtype=np.float32)
	return descrs

def get_frames(im):
	frames = np.asarray(compute_SIFT(im)[0], dtype=np.float32)
	return frames

def hellinger(u, v):
	return np.linalg.norm(np.sqrt(u)-np.sqrt(v), 2)/float(np.sqrt(2))

hellinger_dm = DistanceMetric.get_metric(hellinger)

def plot_matches_corr_eucl(im1, ims, frames1, frames, matches, q):
	## first q images
	ims = ims[:q]
	frames = frames[:q]
	matches = matches[:q]
	N_frames1 = len(frames1)
	plt.figure(figsize=(10, 4.19))
	for k in range(q):
		plt.subplot('1'+str(q)+str(k))
		# plot matches
		plt.imshow(np.concatenate((im1,ims[k]),axis=1))
		for i in range(N_frames1):
			j=matches[k][i,1]
			# plot dots at feature positions
			plt.gca().scatter([frames1[i,0],im1.shape[1]+frames[k][j,0]], [frames1[i,1],frames[k][j,1]], s=5, c=[0,1,0])
			# plot lines
			plt.plot([frames1[i,0],im1.shape[1]+frames[k][j,0]],[frames1[i,1],frames[k][j,1]],linewidth=0.5)
	plt.show()

def plot_matches_lowe_nn(im1, ims, frames1, frames, filtered_matches, q):
	## first q images
	ims = ims[:q]
	filtered_matches = filtered_matches[:q]
	# plot matches
	plt.figure(figsize=(10, 4.19))
	for k in range(q):
		plt.subplot('1'+str(q)+str(k))
		plt.imshow(np.concatenate((im1,ims[k]),axis=1))
		for idx in range(filtered_matches[k].shape[0]):
			    i=filtered_matches[k][idx,0]
			    j=filtered_matches[k][idx,1]
			    # plot dots at feature positions
			    plt.gca().scatter([frames1[i,0],im1.shape[1]+frames[k][j,0]], [frames1[i,1],frames[k][j,1]], s=5, c=[0,1,0]) 
			    # plot lines
			    plt.plot([frames1[i,0],im1.shape[1]+frames[k][j,0]],[frames1[i,1],frames[k][j,1]],linewidth=0.5)
	plt.show()

def plot_matches_ransac(im1, ims, frames1, frames, filtered_matches, q):
	ims = ims[:q]
	frames = frames[:q]
	filtered_matches = filtered_matches[:q]
	# plot matches filtered with RANSAC
	plt.figure(figsize=(10, 4.19))
	for k in range(q):
		plt.subplot('1'+str(q)+str(k))
		plt.imshow(np.concatenate((im1,ims[k]),axis=1))
		for idx in range(filtered_matches[k].shape[0]):
			i=filtered_matches[k][idx,0]
			j=filtered_matches[k][idx,1]
			# plot dots at feature positions
			plt.gca().scatter([frames1[i,0],im1.shape[1]+frames[k][j,0]], [frames1[i,1],frames[k][j,1]], s=5, c=[0,1,0]) 
			# plot lines
			plt.plot([frames1[i,0],im1.shape[1]+frames[k][j,0]],[frames1[i,1],frames[k][j,1]],linewidth=0.5)
	plt.show()

###########
## BoW   ##
###########

def build_KDTree(vocab):
	## source: https://stackoverflow.com/questions/41105806/is-it-possible-to-use-kdtree-with-cosine-similarity
	## Cosine distance ~ Euclidean distance on normalized data
	vocab = (vocab-np.mean(vocab))/np.std(vocab)
	tree = KDTree(vocab, leaf_size=2)#, metric=hellinger_dm)
	return tree

def get_histogram(descrs, tree, num_words):
	N_frames = np.shape(descrs)[0]
	words_fmap = np.zeros((N_frames, 2))
	words_fmap[:, 0] = range(N_frames)
	words_fmap[:, 1] = [idx for lst in tree.query(descrs, k=1)[1] for idx in lst]
	query_hist=np.zeros((num_words, 1))
	unique, counts = np.unique(words_fmap[:, 1], return_counts=True)
	unique = list(map(int, unique))
	query_hist[unique, 0] = counts
	return query_hist

def preprocess_hist(query_hist, tdf_idf):
	for i in range(np.shape(query_hist)[0]):
		query_hist[i, 0] *= tdf_idf[i, 0]
	query_hist = np.sqrt(query_hist)
	query_hist /= np.linalg.norm(query_hist)
	return query_hist

def compute_score(query_hist, hist_i, num_words):
	ps = query_hist*hist_i.T
	n1 = np.linalg.norm(query_hist)
	n2 = np.linalg.norm(hist_i)
	return(ps/(n1*n2))

# fmap: deconvoluted feature map
# images: list of image files to which the feature map should be compared
# num_words: number of visual words to find with K-means
def bow_comparison(fmap, images_list, name="cats", num_words=10, fmap_name="1", list_img=[]):
	print("** BAG OF WORD ANALYSIS **")
	print("* Start")
	name = "bow/bow_" + name
	if (not os.path.exists(folder + name + "_histograms.dat")):
		if (not os.path.exists(folder + name + "_descrs.dat")):
			descrs_list = list(map(get_descrs, images_list))
			for i in range(len(images_list)):
				np.savetxt(folder + name + "_descrs"+str(i)+".dat", descrs_list[i], delimiter=',')
		else:
			descrs_list = [np.loadtxt(folder + name + "_descrs"+str(i)+".dat", delimiter=',') for i in range(len(images_list))]
		print("* Loaded descriptors")
		if (not os.path.exists(folder + name + "_vocab.dat")):
			descrs_mat = np.concatenate(descrs_list)
			kmeans = MiniBatchKMeans(n_clusters=num_words, random_state=0, batch_size=6).fit(descrs_mat)
			vocab = kmeans.cluster_centers_
			print("#words = ", num_words, ";shape vocab = ", np.shape(vocab))
			np.savetxt(folder + name + "_vocab.dat", vocab, delimiter=',')
		else:
			vocab = np.loadtxt(folder + name + "_vocab.dat", delimiter=',')
		print("* Got visual words")
		tree = build_KDTree(vocab)
		g_h = lambda d : get_histogram(d, tree, num_words)
		histograms = list(map(g_h, descrs_list))
		print("* Computed histograms")
		h = histograms[0]
		for i in range(1, len(histograms)):
			h += histograms[i]
		tdf_idf = h/np.sum(h)
		histograms = list(map(lambda hist : np.matrix(preprocess_hist(hist, tdf_idf)).T, histograms))
		histograms = np.concatenate(histograms)
		np.savetxt(folder + name + "_histograms.dat", histograms, delimiter=',')
		np.savetxt(folder + name + "_tdf_idf.dat", tdf_idf, delimiter=',')
		print("* Computed TDF-IDF coefficients")
	else:
		descrs_list = [np.loadtxt(folder + name + "_descrs"+str(i)+".dat", delimiter=',') for i in range(len(images_list))]
		vocab = np.loadtxt(folder + name + "_vocab.dat", delimiter=',')
		tree = build_KDTree(vocab)
		g_h = lambda d : get_histogram(d, tree, num_words)
		histograms = np.loadtxt(folder + name + "_histograms.dat", delimiter=',')
		histograms = [np.matrix(h) for h in histograms.tolist()]
		tdf_idf = np.matrix(np.loadtxt(folder + name + "_tdf_idf.dat", delimiter=',')).T
	descrs = get_descrs(fmap)
	query_hist = np.matrix(preprocess_hist(g_h(descrs), tdf_idf)).T
	#plot_bovw(query_hist)
	scores = np.zeros((1, len(descrs_list)))
	scores[0,:] = [compute_score(query_hist, hist_i, num_words) for hist_i in histograms]
	header = ""
	if (list_img):
		for x in list_img:
			header += x + "\n"
	np.savetxt(folder + name + "_" + fmap_name + "_scores.dat", scores, delimiter='\n', header=header)
	m = np.argmax(scores[0, :])
	print("* Maximum score is " + str(round(scores[0, m], 3)))
	plot_bovw_compare(query_hist, histograms[m], scores[0, m])
	plot_image_compare(fmap, images_list[m]/255.)
	plt.show()

## Test
if (False):
	list_img = glob.glob("../data/cats/*.jpg*")
	assert len(list_img) > 0, "Put some images in the ./data/cats folder"
	images_list = [load_input(im_name) for im_name in list_img]
	fmap = load_input("cat7-1.jpg")
	bow_comparison(fmap, images_list, list_img=list_img)

###########################
## Correspondence points ##
###########################

def ransac(frames1,frames2,matches,N_iters=100,dist_thresh=15):
	# initialize
	max_inliers=0
	tnf=None
	# run random sampling
	for it in range(N_iters):
		# pick a random sample
		i = np.random.randint(0,frames1.shape[0])
		x_1,y_1,s_1,theta_1=frames1[i,:]
		j = matches[i,1]
		x_2,y_2,s_2,theta_2=frames2[j,:]
		# estimate transformation
		from math import cos, sin
		theta = theta_2-theta_1
		s = s_2/s_1
		t_x = x_2 - s*(cos(theta)*x_1 - sin(theta)*y_1)
		t_y = y_2 - s*(sin(theta)*x_1 + cos(theta)*y_1)
		# evaluate estimated transformation
		X_1 = frames1[:,0]
		Y_1 = frames1[:,1]
		X_2 = frames2[matches[:,1],0]
		Y_2 = frames2[matches[:,1],1]
		X_1_prime = s*(X_1*np.cos(theta)-Y_1*np.sin(theta))+t_x
		Y_1_prime = s*(X_1*np.sin(theta)+Y_1*np.cos(theta))+t_y
		dist = np.sqrt((X_1_prime-X_2)**2+(Y_1_prime-Y_2)**2)
		inliers_indices = np.flatnonzero(dist<dist_thresh)
		num_of_inliers = len(inliers_indices)
		# keep if best
		if num_of_inliers>max_inliers:
			max_inliers=num_of_inliers
			best_inliers_indices = inliers_indices
			tnf = [t_x,t_y,s,theta]
	return (tnf,best_inliers_indices)

enorm = lambda d1, d2 : np.linalg.norm(d1-d2)
#enorm = lambda d1, d2 : hellinger(d1, d2)

def get_nn_ratio(u):
	# Get distance values of the first two nearest neighbours
	ids = np.partition(u, 1)[0:2]
	# Compute distance ratio
	if (ids[1] == 0):
		# Case unlikely to happen (i.e. the descriptors are exactly the same)
		# means ids[0] = 0 since ids[0] is a distance and
		# is the smallest
		return(1)
	return(ids[0]/ids[1])

# fmap: deconvoluted feature map
# images: list of image files to which the feature map should be compared
def corresp_comparison(fmap, images_list, name="cats", fmap_name="1", type_c="ransac", list_img=[]):
	print("** CORRESPONDENCE POINT ANALYSIS **")
	print("* Start")
	name = "corresp/corresp_" + name
	if (not os.path.exists(folder + name + "_descrs0.dat")):
		descrs_list = list(map(get_descrs, images_list))
		for i in range(len(images_list)):
			np.savetxt(folder + name + "_descrs"+str(i)+".dat", descrs_list[i], delimiter=',')
	else:
		descrs_list = [np.loadtxt(folder + name + "_descrs"+str(i)+".dat", delimiter=',') for i in range(len(images_list))]
	print("* Loaded descriptors")
	if (not os.path.exists(folder + name + "_frames0.dat")):
		frames_list = list(map(get_frames, images_list))
		for i in range(len(images_list)):
			np.savetxt(folder + name + "_frames"+str(i)+".dat", frames_list[i], delimiter=',')
	else:
		frames_list = [np.loadtxt(folder + name + "_frames"+str(i)+".dat", delimiter=',') for i in range(len(images_list))]
	print("* Loaded frames (keypoints)")
	## Correspondance points with Euclidean distance
	frames_fmap, descrs_fmap = get_frames(fmap), get_descrs(fmap)
	frames = [get_frames(images_list[i]) for i in range(len(descrs_list))]
	N_frames1 = np.shape(descrs_fmap)[0]
	matches=[np.zeros((N_frames1,2),dtype=np.int) for _ in range(len(descrs_list))]
	build_arr = lambda d, d_list : np.array([enorm(d, d_list[k, :]) for k in range(np.shape(d_list)[0])])
	for i in range(len(descrs_list)):
		matches[i][:,0]=range(N_frames1)
		matches[i][:,1]=[np.argmin(build_arr(descrs_fmap[j, :], descrs_list[i])) for j in range(np.shape(descrs_fmap)[0])]
	if (type_c == "eucl"):
		plot_matches_corr_eucl(fmap, images_list, frames_fmap, frames, matches, 3)
		filtered_matches = [m for m in matches]
	print("* Performed simple Euclidean comparison")
	## Lowe's NN criterium
	if (type_c == "lowe"):
		ratio=[np.zeros((N_frames1,1),dtype=np.int) for _ in range(len(descrs_list))]
		NN_threshold=0.8
		filtered_matches = []
		for i in range(len(descrs_list)):
			ratio[i][:,0]=[get_nn_ratio(build_arr(descrs_fmap[j, :], descrs_list[i])) for j in range(len(descrs_fmap))]
			filtered_indices = np.flatnonzero(ratio[i]<NN_threshold)
			filtered_matches.append(matches[i][filtered_indices,:])
		plot_matches_lowe_nn(fmap, images_list, frames_fmap, frames, filtered_matches, 3)
		print("* Applied Lowe's Nearest Neighbour Criterium")
	## Ransac
	if (type_c == "ransac"):
		ransacs_idx = [ransac(frames_fmap,frames[i],matches[i])[1] for i in range(len(descrs_list))]
		filtered_matches = [matches[i][ransacs_idx[i],:] for i in range(len(descrs_list))]
		plot_matches_ransac(fmap, images_list, frames_fmap, frames, filtered_matches, 3)
		print("* Applied RANSAC to filter matches")
	print("IMAGE\t\t\t\t#MATCHES\tCONTRIBUTION")
	total = sum(map(len, filtered_matches))
	if (total):
		contributions = []
		for i in range(len(images_list)):
			header = list_img[i] if (list_img) else "Image #"+str(i)
			contrib = len(filtered_matches[i])/float(total)
			contributions.append(contrib)
			print(header + "\t\t" + str(len(filtered_matches[i])) + "\t\t" + str(round(contrib, 2)))
		contributions = np.array(contributions)
		header = ""
		if (list_img):
			for x in list_img:
				header += x + "\n"
		np.savetxt(folder + name + "_" + fmap_name + "_contributions.dat", contributions, delimiter='\n', header=header)
		plot_image_compare(fmap, images_list[np.argmax(contributions)])

## Test
if (False):
	list_img = glob.glob("../data/cats/*.jpg*")
	assert len(list_img) > 0, "Put some images in the ./data/cats folder"
	images_list = [load_input(im_name) for im_name in list_img]
	fmap = load_input("cat7-1.jpg")
	corresp_comparison(fmap, images_list, list_img=list_img)

###########################################
## Harris corner key point repeatability ##
###########################################

## CREDIT: "A combined corner and edge detector", Harris and Stephens, BMVC 1988
## ~ edge/corner tracking method
## Matching edges and corners instead of keypoints of SIFT descriptors
## which might be more robust

## Moravec's corner detector + plot functions
## CREDIT: adapted from https://github.com/ceroma/mc920-labs/blob/master/lab2/detectors/moravec.py
def draw_corners(image, corners_map):
	radius = 1
	plt.imshow(image)
	for corner in corners_map:
		x, y = corner
		if (x % 10 == 0 and y % 10 == 0):
			print("Corner: " + str(corner))
		for dx in range(-radius, radius + 1):
			for dy in range(-radius, radius + 1):
				plt.plot([x+dx], [y+dy], "r.")
	plt.show()

def get_non_zero_row(diff):
	idx = np.where(~(diff==0).all(1))[0]
	if (len(idx.tolist()) == 0):
		diff = np.zeros((3, 1))
		ff = diff
	else:
		idx = idx[0]
		diff = diff[idx, :, :]
		idx = np.where(~(diff==0).all(1))[0][0]
		diff = diff[idx, :]
	diff = np.asarray(diff, dtype=np.uint8)
	return diff

def draw_edges(im, edges):
	pass

## Moravec's corner detector + plot functions
## CREDIT: adapted from https://github.com/ceroma/mc920-labs/blob/master/lab2/detectors/moravec.py
## Adding modifications suggested in "A combined corner and edge detector", Harris and Stephens, BMVC 1988
## Gives the so-called Harris detector
def moravec(image, img_name='', threshold=250, sigma_square=100., k=0.05): #k in 0.04, 0.06
	# https://github.com/hughesj919/HarrisCorner/blob/master/Corners.py
	corners, edges = [], []
	xy_shifts = [(1, 0), (1, 1), (0, 1), (-1, 1)]
	## Modification #2: gaussian window
	y_range = range(1, np.shape(image)[1]-1)
	x_range = range(1, np.shape(image)[0]-1)
	W = np.zeros(np.shape(image))
	for u in x_range:
		for v in y_range:
			W[u, v] = np.exp(-u*u+v*v)
	W /= 2*sigma_square
	## Discrete gradient of I wrt x (resp. y) using Sobel filters
	kernel_x = np.matrix([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=np.uint8)
	kernel_y = np.matrix([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype=np.uint8)
	X, Y = [np.zeros(np.shape(image)) for _ in range(2)]
	for i in range(3):
		X[:,:,i] = convolve(image[:,:,i], kernel_x)
		Y[:,:,i] = convolve(image[:,:,i], kernel_y)
	for y in y_range:
		for x in x_range:
			# Look for local maxima in min(E) above threshold:
			E = 100000
			for shift in xy_shifts:
				## Window
				#W = np.zeros(np.shape(image))
				#W[x, y, :] = 1 
				diff = np.roll(image, -shift[1], axis=1)
				diff = W*(np.roll(diff, -shift[0], axis=0)-image)
				diff = get_non_zero_row(diff)
				########################################
				## Non-explicit weights (/!\ image in uint8)
				#diff = image[x + shift[0], y + shift[1]]-image[x, y]
				#u_test = image[x + shift[0], y + shift[1]]-image[x, y]
				#if (not np.all(diff == u_test)):
				#	raise ValueError
				#u_test = np.dot(u_test.T, u_test)
				#print(diff == u_test, u_test, diff)
				#print(type(u_test), type(diff))
				#if (u_test != diff):
				#	raise ValueError
				########################################
				## Modification #1: almost isotropic responses using 
				## expansions of the 45° shifts
				A, B, C = [get_non_zero_row(u*W) for u in [X*X, Y*Y, X*Y]]
				diff = A*x*x + 2*C*x*y + B*y*y
				diff = np.dot(diff.T, diff)
				## Modification #3: auto-correlation matrix
				#A, B, C = list(map(np.sum, [A, B, C]))
				#M = np.matrix([[A, C], [C, B]], dtype=np.uint8)
				#diff = (A*B-C*C)-k*(A+B)**2
			## Hysteresis/double thresholding
				if diff < E:
					E = diff
			if E > threshold:
				corners.append((x, y))
				edges.append((x, y))
			##R is positive in the corner region, negative in the edge regions,
			##and small in Hit flat region
	print("Processed image: " + img_name)
	draw_corners(image, corners)
	draw_edges(image, edges)
	return corners, edges

# fmap: deconvoluted feature map
# images: list of image files to which the feature map should be compared
def repeatability_harris(fmap, images_list, name="cats", fmap_name="1", list_img=[], type_c="lowe"):
	print("** REPEATABILITY OF EDGE/CORNER DETECTOR **")
	print("* Start")
	name = "harris/harris_" + name
	if (not os.path.exists(folder + name + "_corners0.dat")):
		_list = list(map(lambda i : moravec(images_list[i], list_img[i]), range(len(images_list))))
		corners_list = [x[0] for x in _list]
		edges_list = [x[1] for x in _list]
		for i in range(len(images_list)):
			np.savetxt(folder + name + "_corners"+str(i)+".dat", corners_list[i], delimiter=',')
			np.savetxt(folder + name + "_edges"+str(i)+".dat", edges_list[i], delimiter=',')
	else:
		corners_list = [np.loadtxt(folder + name + "_corners"+str(i)+".dat", delimiter=',') for i in range(len(images_list))]
	print("* Loaded corners and edges")
	## Correspondance of edges/corners with Euclidean distance
	corners_fmap, edges_fmap = moravec(fmap, "input image")
	## Corner
	## TODO Compute distance bottleneck between the two vectors/Hellinger ?
	N_corners1 = len(corners_fmap)
	matches=[np.zeros((N_corners1,2),dtype=np.int) for _ in range(len(corners_list))]
	build_arr = lambda c, c_list : np.array([enorm(c, c_list[k]) for k in range(len(c_list))])
	for i in range(len(corners_list)):
		matches[i][:,0]=range(N_corners1)
		matches[i][:,1]=[np.argmin(build_arr(corners_fmap[j], corners_list[i])) for j in range(len(corners_fmap))]
	if (type_c == "eucl"):
		#plot_matches_corr_eucl(fmap, images_list, frames_fmap, frames, matches, 3)
		filtered_matches = [m for m in matches]
	print("* Performed simple Euclidean comparison")
	## Lowe's NN criterium
	if (type_c == "lowe"):
		ratio=[np.zeros((N_corners1,1),dtype=np.int) for _ in range(len(corners_list))]
		NN_threshold=0.8
		filtered_matches = []
		for i in range(len(corners_list)):
			ratio[i][:,0]=[get_nn_ratio(build_arr(corners_fmap[j], corners_list[i])) for j in range(len(corners_fmap))]
			filtered_indices = np.flatnonzero(ratio[i]<NN_threshold)
			filtered_matches.append(matches[i][filtered_indices,:])
		#plot_matches_lowe_nn(fmap, images_list, frames_fmap, frames, filtered_matches, 3)
		print("* Applied Lowe's Nearest Neighbour Criterium")
	print("IMAGE\t\t\t\t#MATCHES\tCONTRIBUTION")
	total = sum(map(len, filtered_matches))
	if (total):
		contributions = []
		for i in range(len(images_list)):
			header = list_img[i] if (list_img) else "Image #"+str(i)
			contrib = len(filtered_matches[i])/float(total)
			contributions.append(contrib)
			print(header + "\t\t" + str(len(filtered_matches[i])) + "\t\t" + str(round(contrib, 2)))
		contributions = np.array(contributions)
		header = ""
		if (list_img):
			for x in list_img:
				header += x + "\n"
		np.savetxt(folder + name + "_" + fmap_name + "_contributions.dat", contributions, delimiter='\n', header=header)
		plot_image_compare(fmap, images_list[np.argmax(contributions)])

## Test
if (True):
	list_img = glob.glob("../data/cats/*.jpg*")
	assert len(list_img) > 0, "Put some images in the ./data/cats folder"
	images_list = [load_input(im_name) for im_name in list_img]
	fmap = load_input("cat7-1.jpg")
	repeatability_harris(fmap, images_list, list_img=list_img)
