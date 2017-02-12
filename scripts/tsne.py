import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
# from sklearn.cross_validation import train_test_split // @DeprecationWarning: /usr/local/lib/python2.7/dist-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20., so now we use: \/
from sklearn.model_selection import train_test_split

# 
# 1. Get data
#

# @DONE: broken link to the .csv data, here is the site for the training test data that isn't broken: http://rstudio-pubs-static.s3.amazonaws.com/19668_2a08e88c36ab4b47876a589bb1d61c37.html, but you have to read that page
# dataframe_all = pd.read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")

# @DONE: download pml-training.csv manually, then copy to ../data/, as ../data/pml-training.csv
dataframe_all = pd.read_csv("../data/pml-training.csv") # http://rstudio-pubs-static.s3.amazonaws.com/19668_2a08e88c36ab4b47876a589bb1d61c37.html  // @andy:2017-02-12

# dataframe_all = pd.read_csv("http://rstudio-pubs-static.s3.amazonaws.com/19668_2a08e88c36ab4b47876a589bb1d61c37.html﻿")

num_rows = dataframe_all.shape[0]

#s
# 2. Clean data
#
# #missing elements (NaN) as per col
counter_nan = dataframe_all.isnull().sum()#
counter_without_nan = counter_nan[counter_nan==0]
# remove NaN cols (i.e. missing elements)
dataframe_all = dataframe_all[counter_without_nan.keys()]
# remove first 7 cols (i.e. no discriminative data, e.g. headers/descriptions)
dataframe_all = dataframe_all.ix[:,7:]
# list of cols (last is class label)
columns = dataframe_all.columns
print columns


#
# 3. Create feature vecs, x, and scale
#
x = dataframe_all.ix[:,:-1].values # store in 70-dimensional feature vecs, since 70 == features we have, 1 feature = 1 dimension
standard_scaler = StandardScaler()  # instantiates a standardizer object from sklearn, this will mean shift the data (center to zero), and scale it (@TODO: w/e that means, w.r.t "scale") // @DONE: what is this?
x_std = standard_scaler.fit_transform(x)  # shift values to standardize across features

#
# 4. get class labels y, map to numbers
#

# class labels
y = dataframe_all.ix[:,-1].values 

# map to num
class_labels = np.unique(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#
# 5. partition data into training set and test set
#
test_percentage = 0.1
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = test_percentage, random_state = 0)

# t distributed stochastic neighbour embedding (t-SNE) visualization
tsne = TSNE(n_components=2, random_state = 0)

x_test_2d = tsne.fit_transform(x_test) # now we have 70D feature data projected into 2D space

markers = ("s", "d", "o", "^", "v")
color_map = {0:"red", 1:"blue", 2:"lightgreen", 3:"purple", 4:"cyan"}
plt.figure()
for idx, cl in enumerate(np.unique(x_test_2d)):
	plt.scatter(x=x_test_2d[y_test==cl, 0], y=x_test_2d[y_test==cl, 1], c=color_map[idx], marker=markers[idx], label=cl)

# for idx, cl in enumerate(np.unique(x_test_2d)):
# 	plt.scatter(x = x_test_2d[y_test==idx, 0], y = x_test_2d[y_test==idx, 1], c = color_map[cl], marker=markers[cl], label=idx)
plt.ylabel("X in t-SNE")
plt.ylabel("Y in t-SNE")
plt.legend(loc="upper left")
plt.title("t-SNE visualisation of test data")
plt.show()
