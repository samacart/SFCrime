

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

#running on a windows machine, change filepaths as needed

train_df = pd.read_csv('Data/train.csv')
train_df = train_df.drop(train_df[train_df.Y == 90].index)
train_df = train_df.drop(train_df[train_df.X == -120.5].index)

# Feature Names: address, dayofweek
streets = [w for w in train_df['Address'].str.split()]
p = []
for s in streets:
    p.append([i for i in s if i.isupper() and len(i) > 2])

train_df['street'] = p

le = preprocessing.LabelEncoder()
dow = le.fit_transform(train_df.DayOfWeek.values)
dow_orig = le.classes_
addr = le.fit_transform(train_df.Address)
addr_orig = le.classes_
category = le.fit_transform(train_df.Category)
cat_orig = le.classes_
pdist = le.fit_transform(train_df.PdDistrict)
pd_orig = le.classes_
street = le.fit_transform(train_df.street)
street_orig = le.classes_

train_df['crime'] = category
train_df['addr'] = addr
train_df['dow'] = dow
train_df['pd'] = pdist
train_df['street'] = street

dates = pd.DatetimeIndex(train_df.Dates)

train_df['date'] = dates.date
train_df['time'] = dates.time
train_df['hour'] = dates.hour
train_df['minutes'] = dates.minute
train_df['month'] = dates.month
train_df['woy'] = dates.weekofyear
train_df['year'] = dates.year

train_df.ix[train_df.hour <12,'time_of_day'] = "morning"
train_df.ix[train_df.hour >=12,'time_of_day'] = "midday"
train_df.ix[train_df.hour >14,'time_of_day'] = "afternoon"
train_df.ix[train_df.hour >18,'time_of_day'] = "night"

le = preprocessing.LabelEncoder()
tod = le.fit_transform(train_df.time_of_day.values)
tod_orig = le.classes_

train_df['tod'] = tod

# Season
train_df.ix[train_df.month == 12,'seas'] = "winter"
train_df.ix[train_df.month == 1,'seas'] = "winter"
train_df.ix[train_df.month == 2,'seas'] = "winter"
train_df.ix[train_df.month == 3,'seas'] = "spring"
train_df.ix[train_df.month == 4,'seas'] = "spring"
train_df.ix[train_df.month == 5,'seas'] = "spring"
train_df.ix[train_df.month == 6,'seas'] = "summer"
train_df.ix[train_df.month == 7,'seas'] = "summer"
train_df.ix[train_df.month == 8,'seas'] = "summer"
train_df.ix[train_df.month == 9,'seas'] = "fall"
train_df.ix[train_df.month == 10,'seas'] = "fall"
train_df.ix[train_df.month == 11,'seas'] = "fall"

le = preprocessing.LabelEncoder()
season = le.fit_transform(train_df.seas.values)
season_orig = le.classes_

train_df['season'] = season

train_df_new = train_df[['date','year','month', 'woy', 'hour','minutes','time','tod', 'dow','season', 'pd','addr','street','X','Y','crime']]

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
train_data = np.array(train[['year','dow','tod', 'X', 'Y', 'street']])
dev_data = np.array(dev[['year','dow','tod', 'X', 'Y', 'street']])

# # Decision Tree
# dt = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=0)
# dt.fit(train_data, train_labels)
# preds = dt.predict(dev_data)
# accuracy = metrics.accuracy_score(dev_labels,preds)
#
# print '\nDecision Tree'
# print 'Accuracy (a decision tree):', accuracy
#
rfc = RandomForestClassifier()
rfc.fit(train_data,train_labels)
preds = rfc.predict(dev_data)

accuracy = metrics.accuracy_score(dev_labels,preds)
print 'Accuracy (a random forest):', accuracy
#
# abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, learning_rate=0.1)
#
# abc.fit(train_data, train_labels)
# preds = abc.predict(dev_data)
# accuracy = metrics.accuracy_score(dev_labels,preds)
# print 'Accuracy (adaboost with decision trees):', accuracy
