#coding:utf-8

## Adapted from practicals from Andrea Vedaldi and Andrew Zisserman by Gul Varol and Ignacio Rocco (HMW1)

import cyvlfeat
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from cyvlfeat.plot import plotframes
from sklearn.neighbors import KDTree, DistanceMetric
from sklearn.cluster import MiniBatchKMeans
import os

thres = 0.01

# change some default matplotlib parameters
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.dpi'] = 120

# ignore warnings
warnings.filterwarnings('ignore')

folder = "./data/bow_sift_comp/"
if not os.path.exists(folder):
	os.mkdir(folder)

###########
## TOOLS ##
###########

def rgb2gray(rgb):
	return np.float32(np.dot(rgb[...,:3], [0.299, 0.587, 0.114])/255.0)

def plot_bovw(hst, title="mystery image"):
	plt.hist(np.histogram(hst))
	plt.title("Bag-of-Visual-Words with " + title)
	plt.xlabel("Word ID")
	plt.ylabel("\'Count\'")
	plt.show()

def compute_SIFT(im, t=thres):
	return cyvlfeat.sift.sift(rgb2gray(im),peak_thresh=t)

def get_descrs(im, t=thres):
	return np.asarray(compute_SIFT(im, t=thres)[1], dtype=np.float32) 

def get_histogram(descrs, tree, num_words):
	N_frames = len(descrs)
	words_fmap = np.zeros((N_frames, 2))
	words_fmap[:, 0] = range(N_frames)
	words_fmap[:, 1] = [idx for lst in tree.query(descrs, k=1)[1] for idx in lst]
	query_hist=np.zeros((num_words, 1))
	unique, counts = np.unique(words[:, 1], return_counts=True)
	unique = list(map(int, unique))
	query_hist[unique, 0] = counts
	return query_hist

def preprocess_hist(query_hist, tdf_idf):
	query_hist = query_hist*tdf_idf
	query_hist = np.sqrt(query_hist)
	query_hist = query_hist/np.linalg.norm(query_hist)
	return query_hist

def compute_score(query_hist, hist_i, num_words):
	ps = np.dot(query_hist[:, 0], hist_i[:, 0])
	n1 = np.linalg.norm(query_hist)
	n2 = np.linalg.norm(hist_i)
	return(ps/(n1*n2))

def hellinger(hist1, hist2):
	return np.linalg.norm(np.sqrt(hist1)-np.sqrt(hist2), 2)/float(np.sqrt(2))

hellinger_dm = DistanceMetric.get_metric(hellinger)

# Plot the image and compute and plot (resp. possibly several) feature frames 
# for image @im and threshold value (resp. array) @t
def plot_SIFT(im, t=thres, figsize=8):
	try:
		s = len(t)
	except:
		s = 1
	if (s == 1):
		frames_arr = [compute_SIFT(im, t)[0]]
	else:
		frames_arr = [compute_SIFT(im, tv)[0] for tv in t]
	# Make plots larger
	f, ax_ls = plt.subplots(1, s, figsize=(figsize*s,figsize*s))
	if (s == 1):
		ax_ls = [ax_ls]
	# In order to get the plots set horizontally
	for i in range(s):
		plt.sca(ax_ls[i])
		plt.imshow(im)
		plotframes(frames_arr[i], linewidth=1)
	plt.show()

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

# Find first and second nearest neighbour in @d2 in terms of Euclidean distance
# for every descriptor in @d1, and compute distance ratio
# Returns an array of length #@d1 with distance ratio of matching descriptors in @d2
def compute_ratio_euclidean(d1, d2):
    # Compute the Euclidean distance between @d in @d1 and every
    # descriptor in @d2 (function enorm)
    # Get distance ratio of the first two nearest neightbours
    # for each descriptor @d in @d1
    return([get_nn_ratio(enorm(d, d2)) for d in d1])

def plot_matches_corr_eucl(im1, im2, frames1, frames2):
	# plot matches
	plt.imshow(np.concatenate((im1,im2),axis=1))
	for i in range(N_frames1):
		j=matches[i,1]
		# plot dots at feature positions
		plt.gca().scatter([frames1[i,0],im1.shape[1]+frames2[j,0]], [frames1[i,1],frames2[j,1]], s=5, c=[0,1,0])
		# plot lines
		plt.plot([frames1[i,0],im1.shape[1]+frames2[j,0]],[frames1[i,1],frames2[j,1]],linewidth=0.5)
	plt.show()

def plot_matches_lowe_nn(im1, im2, filtered_matches):
	# plot matches
	plt.imshow(np.concatenate((im1,im2),axis=1))
	for idx in range(filtered_matches.shape[0]):
	    i=filtered_matches[idx,0]
	    j=filtered_matches[idx,1]
	    # plot dots at feature positions
	    plt.gca().scatter([frames1[i,0],im1.shape[1]+frames2[j,0]], [frames1[i,1],frames2[j,1]], s=5, c=[0,1,0]) 
	    # plot lines
	    plt.plot([frames1[i,0],im1.shape[1]+frames2[j,0]],[frames1[i,1],frames2[j,1]],linewidth=0.5)
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
	print("Start")
	name = "bow_" + name
	if (not os.path.exists(folder + name + "_histograms.dat")):
		descrs_list = list(map(get_descrs, images_list))
		np.savetxt(folder + name + "_descrs.dat", np.matrix(descrs_list), delimiter=',')
		print("Loaded descriptors")
		kmeans = MiniBatchKMeans(n_clusters=num_words, random_state=0, batch_size=6).fit(np.matrix(descrs_list))
		vocab = kmeans.cluster_centers_
		print(num_words, np.shape(vocab))
		np.savetxt(folder + name + "_vocab.dat", vocab, delimiter=',')
		print("Got visual words")
		tree = KDTree(vocab, leaf_size=2, metric=hellinger_dm)
		g_h = lambda d : get_histogram(d, tree, num_words)
		histograms = list(map(g_h, descrs_list))
		print("Computed histograms")
		tdf_idf = list(reduce(lambda x, y : x+y, histograms))
		tdf_idf /= np.sum(tdf_idf)
		histograms = list(map(lambda hist : preprocess_hist(hist, tdf_idf), histograms))
		np.savetxt(folder + name + "_histograms.dat", np.matrix(histograms), delimiter=',')
		np.savetxt(folder + name + "_tdf_idf.dat", tdf_idf, delimiter=',')
		print("Computed TDF-IDF coefficients")
	else:
		vocab = np.loadtxt(folder + name + "_vocab.dat", delimiter=',')
		tree = KDTree(vocab, leaf_size=2, metric=hellinger_dm)
		g_h = lambda d : get_histogram(d, tree, num_words)
		histograms = np.loadtxt(folder + name + "_histograms.dat", delimiter=',')
		tdf_idf = np.loadtxt(folder + name + "_tdf_idf.dat", delimiter=',')
	descrs = get_descrs(fmap)
	query_hist = preprocess_hist(g_h(descrs), tdf_idf)
	plot_bovw(query_hist])
	scores = np.zeros((1, len(descrs_list)))
	scores[0,:] = [compute_score(query_hist, hist_i, num_words) for hist_i in histograms]
	np.savetxt(folder + name + "_" + fmap_name + "_scores.dat", tdf_idf, delimiter=',')
	m = np.argmax(scores[0, :]))
	print("Maximum score is " + str(scores[0, m]))
	plot_bovw(histograms[m], title="closest image")
	plot_bovw(query_hist)
	scores_sorted_idx = np.argsort(-scores)
	scores_sorted = scores.ravel()[scores_sorted_idx]
	N=3
	top_N_idx = scores_sorted_idx.ravel()[:N]
	for i in range(N):
		# choose subplot
		plt.subplot(int(np.ceil(N/5)),5,i+1)
		# plot
		plt.imshow(images_list[i])
		plt.axis('off')
		plt.title('score %1.2f' % scores_sorted.ravel()[i])

###########################
## Correspondence points ##
###########################

def match_euclidean(d1, d2):
	enorm = lambda d, d2 : [np.linalg.norm(d-dprime) for dprime in d2]
	# Get the index of the descriptor in @d2 closest to @d in @d1
	# minimizing the Euclidean distance
	return([np.argmin(enorm(d, d2)) for d in d1])

# fmap: deconvoluted feature map
# images: list of image files to which the feature map should be compared
def corresp_comparison(fmap, images_list, name="cats", fmap_name="1"):
	print("Start")
	name = "corresp_" + name
	if (not os.path.exists(folder + name + "_descrs.dat")):
		descrs = list(map(get_descrs, images_list))
		np.savetxt(folder + name + "_descrs.dat", np.matrix(descrs), delimiter=',')
		print("Loaded descriptors")
	else:
		descrs = np.loadtxt(folder + name + "_descrs.dat", delimiter=',')
	## Correspondance points with Euclidean distance
	frames_fmap, descrs_fmap = compute_SIFT(fmap)
	frames = [compute_SIFT(images_list[i])[0] for i in range(len(descrs))]
	N_frames1 = len(descrs_fmap)
	#matches=[np.zeros((N_frames1,2),dtype=np.int)]*len(descrs)
	#for i in range(len(descrs)):
	#	matches[i][:,0]=range(N_frames1)
	#	matches[i][:,1]=match_euclidean(descrs_fmap, descrs[i])
	#plot_matches_corr_eucl(fmap, images_list[0], frames_fmap, frames[0])
	## Lowe's NN criterium
	ratio=[np.zeros((N_frames1,1),dtype=np.int)]*len(descrs)
	NN_threshold=0.8
	filtered_indices = []
	matches = []
	for i in range(len(descrs)):
		ratio[i][:,0]=compute_ratio_euclidean(descrs_fmap, descrs[i])
		filtered_indices.append(np.flatnonzero(ratio[i]<NN_threshold))
		matches = matches[filtered_indices,:]
	#plot_matches_lowe_nn(fmap, images_list[0], matches)
	## Ransac
	ransacs_idx = [ransac(frames_fmap,frames[i],matches[i])[1] for i in range(len(descrs))]
	matches = [matches[i][ransacs_idx[i],:] for i in range(len(descrs))]
	#plot_matches_ransac(fmap, images_list[0], matches)

###########################################
## Harris corner key point repeatability ##
###########################################

## "A combined corner and edge detector", Harris and Stephens, BMVC 1988

## TODO 
