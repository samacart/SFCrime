import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')

import datetime
import pandas as pd
import numpy as np
import operator
from sklearn import metrics
import csv
import time
from sklearn.svm import SVC

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

# Gaussian NB
model = SVC()
model.fit(train_data,train_labels)

preds = model.predict(dev_data)
 accuracy = metrics.accuracy_score(dev_labels,preds)

print 'SVC Accuracy: ', accuracy

# # Decision Tree
# dt = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=0)
# dt.fit(train_data, train_labels)
# preds = dt.predict(dev_data)
# accuracy = metrics.accuracy_score(dev_labels,preds)
#
# print '\nDecision Tree'
# print 'Accuracy (a decision tree):', accuracy
#

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
