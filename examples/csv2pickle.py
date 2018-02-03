import csv
from six.moves import cPickle as pickle
import numpy as np

def convert(path_csv, path_pickle):

    x = []
    with open(path_csv,'rb') as f:
        reader = csv.reader(f)
        for line in reader: x.append(line)

    with open(path_pickle,'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)


convert('/Users/rahuliyer/Downloads/Files to email Jesse/Raw data from mechanical turk/emotions_pull_request_status_from_mechnical_turk.csv','/Users/rahuliyer/Downloads/emotions.pickle')