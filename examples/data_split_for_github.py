'''
Split a given dataset into three different datasets: training, validation and
testing.

This is achieved by splitting the given list of sentences into three separate
lists according to either a given ratio (e.g. [0.7, 0.1, 0.2]) or by an
explicit enumeration. The sentences are also tokenised using the given
vocabulary.

Also splits a given list of dictionaries containing information about
each sentence.

An additional parameter can be set 'extend_with', which will extend the given
vocabulary with up to 'extend_with' tokens, taken from the training dataset.
'''
from __future__ import print_function, unicode_literals
import example_helper
import json
import random
from six.moves import cPickle as pickle
from torchmoji.sentence_tokenizer import SentenceTokenizer


file = open('../data/emotions.txt','r')

DATASET = []
INFO_DICTS = []
header = True
i = 1
length = []
for line in file:
    if header:
        header = False
        continue
    else:
        line = line.replace('\n','')
        line = line.replace('*','')
        line = line.replace('"','')
        a = line.split('\t')
        text = a[3]
        if len(text) < 6 or len(text) >= 250:
            continue
        length.append(len(text))
        DATASET.append(text)
        labels_i = []
        for i in range(4,14):
            if a[i] == str(1):
                labels_i.append(True)
            else:
                labels_i.append(False)
        labels = {'label':labels_i}
        INFO_DICTS.append(labels)
    i += 1

size = len(INFO_DICTS)
numbers = list(range(size))
#print(numbers)
random.seed(100)
shuffled_numbers = random.sample(numbers,k=size)
train = shuffled_numbers[:2103]
validation = shuffled_numbers[2103:2403]
test = shuffled_numbers[2403:]


dataset = {
    'info' : INFO_DICTS,
    'texts' : DATASET,
    'val_ind': validation,
    'train_ind':train,
    'test_ind' : test
}

with open('../data/emotion.pickle','wb') as f:
    pickle.dump(dataset, f)

# DATASET = [
#     'I am sentence 0',
#     'I am sentence 1',
#     'I am sentence 2',
#     'I am sentence 3',
#     'I am sentence 4',
#     'I am sentence 5',
#     'I am sentence 6',
#     'I am sentence 7',
#     'I am sentence 8',
#     'I am sentence 9 newword',
#     ]

# INFO_DICTS = [
#     {'label': 'sentence 0'},
#     {'label': 'sentence 1'},
#     {'label': 'sentence 2'},
#     {'label': 'sentence 3'},
#     {'label': 'sentence 4'},
#     {'label': 'sentence 5'},
#     {'label': 'sentence 6'},
#     {'label': 'sentence 7'},
#     {'label': 'sentence 8'},
#     {'label': 'sentence 9'},
#     ]

# with open('../model/vocabulary.json', 'r') as f:
#     vocab = json.load(f)
# st = SentenceTokenizer(vocab, 30)

# # Split using the default split ratio
# splits = st.split_train_val_test(DATASET, INFO_DICTS)

# train_ind = splits[]

# Split explicitly
# print(st.split_train_val_test(DATASET,
#                               INFO_DICTS,
#                               [[0, 1, 2, 4, 9], [5, 6], [7, 8, 3]],
#                               extend_with=1))
