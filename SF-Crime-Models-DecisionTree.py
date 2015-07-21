import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')

import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import operator
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import preprocessing
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import csv
import time

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from matplotlib.colors import ListedColormap

#running on a windows machine, change filepaths as needed

train_df_new = pd.read_csv('Data/train_pp.csv')

# Split this into Dev and Training Data
DEV_SIZE = 0.20

# fix a random seed
np.random.seed(0)
# Create boolean mask
# np.random creates a vector of random values between 0 and 1
# Those values are filtered to create a binary mask
msk = np.random.rand(len(train_df_new)) < DEV_SIZE

dev = train_df_new[msk]
dev_labels = np.array(dev.crime)
dev.drop('crime',1,inplace=True)

train = train_df_new[~msk]  # inverse of boolean mask
train_labels = np.array(train.crime)
train.drop('crime',1,inplace=True)

print "Dev: " + str(dev.shape)
print "Train: " + str(train.shape)

# convert training set to np array
train_data = np.array(train[['year','dow','tod', 'hour', 'X', 'Y', 'street', 'season']])
dev_data = np.array(dev[['year','dow','tod', 'hour','X', 'Y', 'street', 'season']])

# # Decision Tree
# dt = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=0)
# dt.fit(train_data, train_labels)
# preds = dt.predict(dev_data)
# accuracy = metrics.accuracy_score(dev_labels,preds)
#
# print '\nDecision Tree'
# print 'Accuracy (a decision tree):', accuracy
#

km = KMeans(n_clusters=50)
km.fit(dev_data)

cm_bright = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFFFF'])

plt.figure(figsize=(12, 4))

p = plt.subplot(1, 3, 1)
p.scatter(dev_data[:, 1], dev_data[:, 2], c=km.predict(dev_data), cmap=cm_bright)
plt.title('KMeans')

plt.show()

# rfc = RandomForestClassifier()
# rfc.fit(train_data,train_labels)
# preds = rfc.predict(dev_data)
#
# accuracy = metrics.accuracy_score(dev_labels,preds)
# print 'Accuracy (a random forest):', accuracy

# abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, learning_rate=0.1)
#
# abc.fit(train_data, train_labels)
# preds = abc.predict(dev_data)
# accuracy = metrics.accuracy_score(dev_labels,preds)
# print 'Accuracy (adaboost with decision trees):', accuracy
