#coding: utf-8

import glob
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

## PCA on contributions of each training image to reconstructed input

## Load vectors associated with each reconstructed input
contributions_ = glob.glob("./contributions/harris_siamese_*_contributions.dat")
order = [int(c.split("_")[2]) for c in contributions_]
ra = sorted(range(len(contributions_)), key=lambda x : order[x])
contributions_ = [contributions_[i] for i in ra]
order = [order[i] for i in ra]
contributions = np.matrix([np.loadtxt(c).tolist() for c in contributions_])
print(np.shape(contributions))
#print(contributions)
means = list(map(lambda x : round(np.mean(x), 2), [contributions[:,i] for i in range(len(ra))]))
stds = list(map(lambda x : round(np.std(x), 2), [contributions[:,i] for i in range(len(ra))]))

plt.errorbar(range(len(means)), means, yerr=stds, fmt='o')
plt.title("Contributions (in terms of matches) of each training\nimage to the reconstructed input")
plt.xlabel("Label of training image")
plt.ylabel("Contributions (average % of matches on reconstructed input + std)")
plt.show()
