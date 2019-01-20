#coding: utf-8

import sys
import argparse
import glob
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

parser = argparse.ArgumentParser(description='Plot/Results of deconvoluted images')
parser.add_argument('--tmodel', type=str, default='bow', metavar='O',
                    help='Model: [\'conv\', \'vonc\', \'vgg\'].')
parser.add_argument('--tmethod', type=str, default='bow', metavar='M',
                    help='Method: [\'bow\', \'sift\', \'harris\'].')
parser.add_argument('--tdata', type=str, default='siamese', metavar='D',
                    help='Method: [\'siamese\', \'cats\'].')
args = parser.parse_args()

if (args.tmethod != "bow"):
	## Plot contributions of each training image to reconstructed input

	## Load vectors associated with each reconstructed input
	## Should be called from the root folder
	name = "corresp" if (args.tmethod == "sift") else args.tmethod
	contributions_ = glob.glob("./slides+report/contributions/"+name+"_"+args.tdata+"_"+args.tmodel+"_*_a_contributions.dat")
	contributions = np.concatenate([np.matrix(np.loadtxt(c)).T for c in contributions_], axis=1)
	m, n = np.shape(contributions)
	k = min(11, m)
	means = np.array(list(map(lambda x : round(np.mean(x), 2), [contributions[i,:] for i in range(m)])))
	stds = np.array(list(map(lambda x : round(np.std(x), 2), [contributions[i, :] for i in range(m)])))
	## Plot the k highest contributions
	highest = means.argsort()[-k:]
	print(highest, "Images which have the " + str(k) + " highest contributions")
	means = means[highest].tolist()
	stds = stds[highest].tolist()

	plt.errorbar(range(len(means)), means, yerr=stds, fmt='o')
	plt.title("Contributions (%matches) of each training\nimage to the reconstructed input ranked in increasing value order")
	plt.xlabel("Number of corresponding training image")
	plt.xticks(range(len(means)), highest)
	plt.ylabel("Contributions (average % of matches\non reconstructed input + std)")
	plt.show()
else:
	## Should be called from the root folder
	scores_ = glob.glob("./slides+report/figures_/bow_analysis/bow_"+args.tdata+"_"+args.tmodel+"_*_scores.dat")
	before_train = list(filter(lambda x : "b" in x.split("_"), scores_))
	after_train = list(filter(lambda x : "a" in x.split("_"), scores_))
	bscores = np.array([np.max(np.loadtxt(s)).tolist() for s in before_train])
	ascores = np.array([np.max(np.loadtxt(s)).tolist() for s in after_train])
	a, b = list(map(np.median, [bscores, ascores]))
	c, d = list(map(np.mean, [bscores, ascores]))
	scoresm = np.matrix([[a, c], [b, d]])
	print("Median max. score | Mean max. score"+"// before training | after training")
	print(scoresm)
	np.savetxt(X=scoresm, fname="./slides+report/scores_med_mean_ab_train_"+args.tmodel+"_"+args.tdata+".csv", delimiter=",")
