#!/usr/bin/env python

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import operator
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB

def draw_plot(df, title):
    bar = df.plot(kind='barh',
                  title=title,
                  fontsize=8,
                  figsize=(12,8),
                  stacked=False,
                  width=1
    )

    plt.show()

def visualize_data(df, column, title, items=0):
    df.columns     = df.columns.map(operator.methodcaller('lower'))
    by_col         = df.groupby(column)
    col_freq       = by_col.size()

    col_freq.sort(ascending=True, inplace=True)
    draw_plot(col_freq[slice(-1, - items, -1)], title)

if __name__ == '__main__':
    # Read in the Training Data
    train_df = pd.read_csv('Data/train.csv')
    test_df = pd.read_csv('Data/test.csv')

    # Visualize the entire dataset
    # visualize_data(train_df, 'category',   'Top Crime Categories')
    # visualize_data(train_df, 'resolution', 'Top Crime Resolutions')
    # visualize_data(train_df, 'pddistrict', 'Police Department Activity')
    # visualize_data(train_df, 'dayofweek',  'Days of the Week')
    # visualize_data(train_df, 'address',    'Top Crime Locations', items=20)
    # visualize_data(train_df, 'descript',   'Descriptions', items=20)


    # Split this into Dev and Training Data
    DEV_SIZE = 0.20

    # Create boolean mask
    # np.random creates a vector of random values between 0 and 1
    # Those values are filtered to create a binary mask
    msk = np.random.rand(len(train_df)) < DEV_SIZE

    dev = train_df[msk]
    dev_labels = dev.Category.values
    dev = dev.drop(['Category','Dates','Descript','X','Y','Resolution'], axis=1)

    train = train_df[~msk]  # inverse of boolean mask
    train_labels = train.Category.values
    train = train.drop(['Category','Dates','Descript','X','Y','Resolution'], axis=1)

    print "Dev: " + str(dev.shape)
    print "Train: " + str(train.shape)
    print "Test: " + str(test_df.shape)

    # Target Names are the categories
    print train_labels

    # Feature Names: address, dayofweek
    days = {}
    cnt=0
    for i in np.unique(dev.DayOfWeek.values):
        days[i] = cnt
        cnt+=1

    dict ={'DayOfWeek' : days}

    address = {}
    cnt=0
    for i in np.unique(dev.Address.values):
        address[i] = cnt
        cnt+=1

    dict['Address'] = address

    dev = dev.replace(dict)

    model = GaussianNB()
    model.fit(dev,dev_labels)

    predicted = np.array(model.predict_proba(train))
    labels = ['Id']
    for i in model.classes_:
        labels.append(i)

    fo = csv.writer('GaussianNB.csv', lineterminator='\n')
    fo.writerow(labels)

    for i, pred in enumerate(predicted):
        fo.writerow([i] + list(pred))
