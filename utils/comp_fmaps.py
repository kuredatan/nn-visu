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
from cyvlfeat.plot import plotframes
from PIL import Image
from scipy.misc import imresize

sz = 200

thres = 0.01

# change some default matplotlib parameters
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.dpi'] = 120

# ignore warnings
warnings.filterwarnings('ignore')

ff = "/home/reda/Projets20182019/Vision/Projet/interpretation/DeconvNet/"
folder = ff+"data/bow_sift_comp/"
if not os.path.exists(folder):
	os.mkdir(folder)

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
	plt.title("Bag-of-Visual-Words with mystery image")
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
	plt.title("Mystery image")
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

def stack_mat(descrs_list):
	mat = np.concatenate(descrs_list)
	return mat

def hellinger(u, v):
	return np.linalg.norm(np.sqrt(u)-np.sqrt(v), 2)/float(np.sqrt(2))

hellinger_dm = DistanceMetric.get_metric(hellinger)

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

# Plot the image and compute and plot (resp. possibly several) feature frames 
# for image @im
def plot_SIFT(im, figsize=8):
	frames = compute_SIFT(im)[0]
	img=cv2.drawKeypoints(im,frames,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imwrite('sift_keypoints.jpg',img)

def plot_matches_corr_eucl(im1, ims, frames1, frames, matches, q):
	## first q images
	ims = ims[:q]
	frames = frames[:q]
	matches = matches[:q]
	N_frames1 = len(frames1)
	n = len(frames)
	plt.figure(figsize=(10, 4.19))
	for k in range(q):
		plt.subplot('1'+str(n)+str(k))
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
	for k in range(q):
		plt.imshow(np.concatenate((im1,ims[k]),axis=1))
		for idx in range(filtered_matches[k].shape[0]):
			    i=filtered_matches[k][idx,0]
			    j=filtered_matches[k][idx,1]
			    # plot dots at feature positions
			    plt.gca().scatter([frames1[i,0],im1.shape[1]+frames[k][j,0]], [frames1[i,1],frames[k][j,1]], s=5, c=[0,1,0]) 
			    # plot lines
			    plt.plot([frames1[i,0],im1.shape[1]+frames[k][j,0]],[frames1[i,1],frames[k][j,1]],linewidth=0.5)
	plt.show()

def plot_matches_ransac(im1, im2, filtered_matches):
	# plot matches filtered with RANSAC
	plt.imshow(np.concatenate((im1,im2),axis=1))
	for idx in range(filtered_matches.shape[0]):
		i=filtered_matches[idx,0]
		j=filtered_matches[idx,1]
		# plot dots at feature positions
		plt.gca().scatter([frames1[i,0],im1.shape[1]+frames2[j,0]], [frames1[i,1],frames2[j,1]], s=5, c=[0,1,0]) 
		# plot lines
		plt.plot([frames1[i,0],im1.shape[1]+frames2[j,0]],[frames1[i,1],frames2[j,1]],linewidth=0.5)
	plt.show()

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

###########
## BoW   ##
###########

# fmap: deconvoluted feature map
# images: list of image files to which the feature map should be compared
# num_words: number of visual words to find with K-means
def bow_comparison(fmap, images_list, name="cats", num_words=10, fmap_name="1"):
	print("* Start")
	name = "bow_" + name
	if (not os.path.exists(folder + name + "_histograms.dat")):
		if (not os.path.exists(folder + name + "_descrs.dat")):
			descrs_list = list(map(get_descrs, images_list))
			for i in range(len(images_list)):
				np.savetxt(folder + name + "_descrs"+str(i)+".dat", descrs_list[i], delimiter=',')
		else:
			descrs_list = [np.loadtxt(folder + name + "_descrs"+str(i)+".dat", delimiter=',') for i in range(len(images_list))]
		print("* Loaded descriptors")
		if (not os.path.exists(folder + name + "_vocab.dat")):
			descrs_mat = stack_mat(descrs_list)
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
	np.savetxt(folder + name + "_" + fmap_name + "_scores.dat", tdf_idf, delimiter=',')
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
	bow_comparison(fmap, images_list)

###########################
## Correspondence points ##
###########################

enorm = lambda d, d_list : [np.linalg.norm(d-dprime) for dprime in d_list]
#enorm = lambda d, d_list : [hellinger(d, dprime) for dprime in d_list]

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

# Find first and second nearest neighbour in @d_list in terms of a given distance
# for every descriptor in @d1, and compute distance ratio
# Returns an array of length #@d1 with distance ratio of matching descriptors in @d_list
def compute_ratio_euclidean(d1, d_list):
	# Compute the distance between @d in @d1 and every
	# descriptor in @d_list
	# Get distance ratio of the first two nearest neightbours
	# for each descriptor @d in @d1
	return([get_nn_ratio(enorm(d, d_list)) for d in d1])

def match_euclidean(d1, d_list):
	# Get the index of the descriptor in @d_list closest to @d in @d1
	# minimizing the distance
	return([np.argmin(enorm(d, d_list)) for d in d1])

# fmap: deconvoluted feature map
# images: list of image files to which the feature map should be compared
def corresp_comparison(fmap, images_list, name="cats", fmap_name="1", type_c="lowe"):
	print("* Start")
	name = "corresp_" + name
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
	frames_fmap, descrs_fmap = compute_SIFT(fmap)
	frames = [compute_SIFT(images_list[i])[0] for i in range(len(descrs_list))]
	N_frames1 = np.shape(descrs_fmap)[0]
	matches=[np.zeros((N_frames1,2),dtype=np.int)]*len(descrs_list)
	for i in range(len(descrs_list)):
		matches[i][:,0]=range(N_frames1)
		matches[i][:,1]=match_euclidean(descrs_fmap, descrs_list[i])
	if (type_c == "eucl"):
		## TODO correction
		plot_matches_corr_eucl(fmap, images_list, frames_fmap, frames, matches, 3)
		print("* Performed simple Euclidean comparison")
	## Lowe's NN criterium
	if (type_c == "lowe"):
		ratio=[np.zeros((N_frames1,1),dtype=np.int)]*len(descrs_list)
		NN_threshold=0.8
		filtered_matches = []
		for i in range(len(descrs_list)):
			ratio[i][:,0]=compute_ratio_euclidean(descrs_fmap, descrs_list[i])
			filtered_indices = np.flatnonzero(ratio[i]<NN_threshold)
			filtered_matches.append(matches[i][filtered_indices,:])
		plot_matches_lowe_nn(fmap, images_list[1:2], frames_fmap, frames[1:2], filtered_matches[1:2], 1)
	## Ransac
	if (type_c == "ransac"):
		ransacs_idx = [ransac(frames_fmap,frames[i],matches[i])[1] for i in range(len(descrs_list))]
		matches = [matches[i][ransacs_idx[i],:] for i in range(len(descrs_list))]
		plot_matches_ransac(fmap, images_list[0], matches[0])

## Test
list_img = glob.glob("../data/cats/*.jpg*")
assert len(list_img) > 0, "Put some images in the ./data/cats folder"
images_list = [load_input(im_name) for im_name in list_img]
fmap = load_input("cat7-1.jpg")
corresp_comparison(fmap, images_list)

###########################################
## Harris corner key point repeatability ##
###########################################

## "A combined corner and edge detector", Harris and Stephens, BMVC 1988

## TODO 
