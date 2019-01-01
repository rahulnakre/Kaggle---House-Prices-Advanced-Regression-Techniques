# used for manipulating directory paths
import os
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns

# PREPROCESSING TUT FROM
#https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
train = pd.read_csv("train.csv")

# IMPORTANT PRACTISE: duplicate check
ids_unique = len(set(train.Id))
ids_total = train.shape[0]
ids_duplicate  = ids_total - ids_unique
print("there are " + str(ids_duplicate) + " dulplicate Ids out of " 
      + str(ids_total))
# ***** why drop Ids column?
train.drop("Id", axis = 1, inplace=True)

# check out general corr matrix to understand what goin on 
corr_mat = train.corr()
#sns.heatmap(corr_mat)
# pd.nlargest() returns the largest n rows of the specified column
n = 10
cols = corr_mat.nlargest(n, "SalePrice")["SalePrice"]
#sns.distplot(cols)

# store train[cols.index] cus we only want those n observations
# .T cus we want the SalePrice, OverallQual, etc, to be the rows
# because np.corrcoeff needs it to be like that
t = train[cols.index].values.T
# correlation matrix for the n columns we got earlier
cm = np.corrcoef(t) 
sns.set(font_scale=1.25)
heat_map = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                       annot_kws={'size': 10}, yticklabels=cols.index.values,
                       xticklabels=cols.index.values)


# timeseries analysis 
# year built and saleprice comparison
ts_sale = train[["YearBuilt", "SalePrice"]]


#
'''
# try to notice outliers w regards to GrLivArea (Above Ground Living Area)
plt.scatter(train["GrLivArea"], train["SalePrice"], c="blue", marker="s",
           alpha=0.1)
plt.title("Outlier Search")
plt.xlabel("Above Ground Living Area Sq. Ft (GrLivArea)")
plt.ylabel("SalePrice")
'''
'''
# try to notice outliers w regards to GrLivArea (Above Ground Living Area)
plt.scatter(train["LotArea"], train["SalePrice"], c="blue",
            alpha=0.1)
plt.title("Outlier Search")
plt.xlabel("Lot Areaa (Sq.ft)")
plt.ylabel("SalePrice")
'''
# noticed that stuff above 4000 seems to be an outlier
# train = train[train["GrLivArea"] < 4000]


