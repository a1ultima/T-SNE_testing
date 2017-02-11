import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

# 
# 1. Get data
#

# @DONE: broken link to the .csv data, here is the site for the training test data that isn't broken: http://rstudio-pubs-static.s3.amazonaws.com/19668_2a08e88c36ab4b47876a589bb1d61c37.html, but you have to read that page
# dataframe_all = pd.read_csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")

# @DONE: download pml-training.csv manually, then copy to ../data/, as ../data/pml-training.csv
dataframe_all = pd.read_csv("../data/pml-training.csv")
num_rows = dataframe_all.shape[0]

#
# 2. Clean data
#
# #missing elements (NaN) as per col
counter_nan = dataframe_all.isnull().sum()#
counter_without_nan = counter_nan[counter_nan==0]
# remove NaN cols (i.e. missing elements)
dataframe_all = dataframe_all[counter_without_nan.keys()]
# remove first 7 cols (i.e. no discriminative data, e.g. headers/descriptions)
dataframe_all = dataframe_all.ix[:,7:]

#
# 3. Create feature vecs
#
x = dataframe_all.ix[:,:-1]