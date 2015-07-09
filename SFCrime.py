#!/usr/bin/env python

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import operator

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

def make_sets(data_df, test_portion):
    import random as rnd

    tot_ix = range(len(data_df))
    test_ix = np.sort(rnd.sample(tot_ix, int(test_portion * len(data_df))))
    train_ix = list(set(tot_ix) ^ set(test_ix))

    test_df = data_df.ix[test_ix]
    train_df = data_df.ix[train_ix]

    return train_df, test_df

if __name__ == '__main__':
    # Read in the Training Data
    df = pd.read_csv('Data/train.csv')

    # Split this into Dev and Training Data
    dev, train = make_sets(df, 0.2)

    # Visualize the entire dataset
    visualize_data(df, 'category',   'Top Crime Categories')
    visualize_data(df, 'resolution', 'Top Crime Resolutions')
    visualize_data(df, 'pddistrict', 'Police Department Activity')
    visualize_data(df, 'dayofweek',  'Days of the Week')
    visualize_data(df, 'address',    'Top Crime Locations', items=20)
    visualize_data(df, 'descript',   'Descriptions', items=20)
