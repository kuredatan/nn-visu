#coding:utf-8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from sklearn.neighbors import KDTree, DistanceMetric
from sklearn.cluster import MiniBatchKMeans, KMeans
import os
import glob
import cyvlfeat
from cyvlfeat.plot import plotframes
from PIL import Image
from scipy.misc import imresize
import plot_matching as plm

sz = 100

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
	sift = cyvlfeat.sift.sift(plm.rgb2grey(im),peak_thresh=thres,compute_descriptor=True)
	return sift

def get_descrs(im):
	descrs = np.asarray(compute_SIFT(im)[1], dtype=np.float32)
	return descrs

def hellinger(u, v):
	return np.linalg.norm(np.sqrt(u)-np.sqrt(v), 2)/float(np.sqrt(2))

hellinger_dm = DistanceMetric.get_metric(hellinger)

def results_correspondences(inliers, list_img, fname, matches):
	n = len(list_img)
	contributions = np.array(list(map(lambda x: len(np.nonzero(x)[0]), inliers)))
	## Sum of all inliers
	total = np.sum(contributions)
	if (total):
		contributions = contributions/np.float32(total)
		print("IMAGE\t\t\t\t#MATCHES\tCONTRIBUTIONS")
		for i in range(n):
			header = list_img[i] if (list_img) else "Image #"+str(i)
			print(header + "\t\t" + str(int(contributions[i]*total)) + "\t\t" + str(round(contributions[i], 2)))
		header = ""
		if (list_img):
			for x in list_img:
				header += x + "\n"
		np.savetxt(fname + "_contributions.dat", contributions, delimiter='\n', header=header)
		m = np.argmax(contributions)
		src, dst = matches[m]
		return m
	else:
		print("No inliers!!")
	return None

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
			if (np.shape(descrs_mat)[0] > 1000):
				kmeans = MiniBatchKMeans(n_clusters=num_words, random_state=0, batch_size=6).fit(descrs_mat)
			else:
				kmeans = KMeans(n_clusters=num_words, random_state=0).fit(descrs_mat)
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

###########################
## Correspondence points ##
###########################

## Distance
#enorm = lambda d1, d2 : np.linalg.norm(d1-d2)
enorm = lambda d1, d2 : hellinger(d1, d2)

## SOURCE: practicals from Andrea Vedaldi and Andrew Zisserman by Gul Varol and Ignacio Rocco (HMW1)
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

# fmap: deconvoluted feature map
# images: list of image files to which the feature map should be compared
def corresp_comparison(fmap, images_list, name="cats", fmap_name="1", list_img=[]):
	print("** CORRESPONDENCE POINT ANALYSIS **")
	print("* Start")
	name = "corresp/corresp_" + name
	n = len(images_list)
	if (not os.path.exists(folder + name + "_descrs0.dat")):
		_list = list(map(compute_SIFT, images_list))
		descrs_list = list(map(lambda x : x[1], _list))
		frames = list(map(lambda x : x[0], _list))
		for i in range(n):
			np.savetxt(folder + name + "_descrs"+str(i)+".dat", descrs_list[i], delimiter=',')
			np.savetxt(folder + name + "_frames"+str(i)+".dat", frames[i], delimiter=',')
	else:
		descrs_list = [np.loadtxt(folder + name + "_descrs"+str(i)+".dat", delimiter=',') for i in range(n)]
		frames = [np.loadtxt(folder + name + "_frames"+str(i)+".dat", delimiter=',') for i in range(n)]
	print("* Loaded descriptors and keypoints")
	## Correspondance points
	frames_fmap, descrs_fmap = compute_SIFT(fmap)
	N_frames1 = np.shape(descrs_fmap)[0]
	matches=[np.zeros((N_frames1,2),dtype=np.int) for _ in range(n)]
	## Return index of closest descriptors in terms of distance enorm for each fmap descriptor d
	build_arr = lambda d, d_list : np.array([enorm(d, d_list[k, :]) for k in range(np.shape(d_list)[0])])
	inliers = []
	nu = np.shape(descrs_fmap)[0]
	for i in range(n):
		matches[i][:,0]=range(N_frames1)
		matches[i][:,1]=[np.argmin(build_arr(descrs_fmap[j, :], descrs_list[i])) for j in range(nu)]
	for i in range(n):
		# robustly estimate affine transform model with RANSAC
		ransacs_idx = [ransac(frames_fmap,frames[i],matches[i])[1] for i in range(n)]
		inliers = [matches[i][ransacs_idx[i],:] for i in range(n)]
		src_dst = [[frames_fmap[inliers[i][:, 0], :2], frames[i][inliers[i][:, 1], :2]] for i in range(n)]
	fname = folder + name + "_" + fmap_name
	matches = src_dst
	m = results_correspondences(inliers, list_img, fname, matches)
	if (m):
		from skimage.feature import plot_matches
		fig, ax = plt.subplots(nrows=1, ncols=1)
		plot_matches(ax, fmap, images_list[m], frames_fmap[:, :2], frames[m][:, :2], inliers[m])
		plt.show()

###########################################
## Harris corner key point repeatability ##
###########################################

## CREDIT: "A combined corner and edge detector", Harris and Stephens, BMVC 1988
## ~ corner tracking method
## Matching corners instead of keypoints of SIFT descriptors
## which might be more robust

## SOURCE: http://scikit-image.org/docs/0.5/auto_examples/plot_harris.html
def plot_harris_points(image, filtered_coords):
	plt.plot()
	plt.imshow(image)
	plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords],'b.')
	plt.axis('off')
	plt.title("Harris corner points")
	plt.show()

# fmap: deconvoluted feature map
# images: list of image files to which the feature map should be compared
# SOURCE: http://scikit-image.org/docs/dev/auto_examples/transform/plot_matching.html#sphx-glr-auto-examples-transform-plot-matching-py
def repeatability_harris(fmap, images_list, name="cats", fmap_name="1", list_img=[]):
	print("** REPEATABILITY OF CORNER DETECTOR **")
	print("* Start")
	name = "harris/harris_" + name
	n = len(images_list)
	if (not os.path.exists(folder + name + "_corners0.dat")):
		corners_list = list(map(plm.extract_corners, images_list))
		for i in range(n):
			np.savetxt(folder + name + "_corners"+str(i)+".dat", corners_list[i], delimiter=',')
			#plot_harris_points(images_list[i], corners_list[i])
	else:
		corners_list = [np.loadtxt(folder + name + "_corners"+str(i)+".dat", delimiter=',') for i in range(n)]
	print("* Loaded corners")
	## Correspondance of corners
	corners_fmap = plm.extract_corners(fmap)
	## Using RANSAC
	matches = [plm.find_correspondences(fmap, images_list[i], corners_fmap, corners_list[i]) for i in range(n)]
	fname = folder + name + "_" + fmap_name
	inliers = [m[0] for m in matches]
	matches = [m[1:] for m in matches]
	m = results_correspondences(inliers, list_img, fname, matches)
	if (m):
		plm.plot_correspondences(fmap, images_list[m], matches[m][0], matches[m][1], inliers[m])

###################################################################

## Tests
list_img = glob.glob("../data/cats/*.jpg*")
assert len(list_img) > 0, "Put some images in the ./data/cats folder"
images_list = [load_input(im_name) for im_name in list_img]
fmap = load_input("cat7-1.jpg")
if (False):
	bow_comparison(fmap, images_list, list_img=list_img)
if (False):
	corresp_comparison(fmap, images_list, list_img=list_img)
if (False):
	repeatability_harris(fmap, images_list, list_img=list_img)
