import csv
from six.moves import cPickle as pickle
import numpy as np


def convert(path_pickle,path_csv):

    x = []
    with open(path_pickle,'rb') as f:
        x = pickle.load(f)
    print(x)
    exit()
    with open(path_csv,'w') as f:
    	f.write(str(x))

convert('/Users/rahuliyer/work/sentiment_analysis/torchMoji/data/SE0714/raw.pickle','/Users/rahuliyer/work/sentiment_analysis/torchMoji/data/SE0714/raw_csv.csv')