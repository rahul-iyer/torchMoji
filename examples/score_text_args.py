# -*- coding: utf-8 -*-

""" Use torchMoji to score texts for emoji distribution.

The resulting emoji ids (0-63) correspond to the mapping
in emoji_overview.png file at the root of the torchMoji repo.

Writes the result to a csv file.
"""
from __future__ import print_function, division, unicode_literals
import example_helper
import json
import csv
import numpy as np
import argparse
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

OUTPUT_PATH = 'test_sentences.csv'

# TEST_SENTENCES = ['I love mom\'s cooking',
#                   'I love how you never reply back..',
#                   'I love cruising with my homies',
#                   'I love messing with yo mind!!',
#                   'I love you and now you\'re just gone..',
#                   'This is shit',
#                   'This is the shit',
#                   "Let's keep that in the README for now, Contributing.md is more a guideline for pull requests. At that step, it's probable that contributors will have already tested their changes (I hope)."]

parser = argparse.ArgumentParser(description='score_texts_args.py')

parser.add_argument('-t', default='Hi, How are you?',
                    help="""Input text""")

opt = parser.parse_args() 

TEST_SENTENCES = []
TEST_SENTENCES.append(opt.t)

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

maxlen = 30

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)

st = SentenceTokenizer(vocabulary, maxlen)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = torchmoji_emojis(PRETRAINED_PATH)
print(model)
print('Running predictions.')
tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
prob = model(tokenized)

# Mappings
mappings = {
    '35'    : 'sadness',
    '5'     : 'sadness',
    '27'    : 'sadness',
    '43'    : 'sadness',
    '45'    : 'sadness',
    '52'    : 'sadness',
    '2'     : 'sadness',
    '29'    : 'sadness',
    '3'     : 'sadness',
    '34'    : 'sadness',
    '46'    : 'sadness',
    '37'    : 'anger',
    '32'    : 'anger',
    '55'    : 'anger',
    '22'    : 'dissatisfaction',
    '25'    : 'dissatisfaction',
    '1'     : 'dissatisfaction',
    '19'    : 'dissatisfaction',
    '0'     : 'happy',
    '51'    : 'unknown',
    '62'    : 'awkwardness',
    '12'    : 'awkwardness',
    '20'    : 'awkwardness',
    '14'    : 'avoidance',
    '39'    : 'avoidance',
    '42'    : 'unknown',
    '57'    : 'strength',
    '58'    : 'strength',
    '30'    : 'acheivement',
    '13'    : 'acheivement',
    '38'    : 'disagreement',
    '56'    : 'disagreement',
    '4'     : 'happy',
    '36'    : 'happy',
    '10'    : 'happy',
    '7'     : 'happy',
    '53'    : 'happy',
    '6'     : 'appreciation',
    '33'    : 'appreciation',
    '17'    : 'appreciation',
    '40'    : 'appreciation',
    '50'    : 'humour', #happy 
    '9'     : 'humour', #happy
    '54'    : 'humour', #happy
    '31'    : 'happy', # need for check
    '44'    : 'happy', # need for check
    '15'    : 'cheeky',
    '26'    : 'cheeky',
    '11'    : 'happy', # but when listening to music
    '48'    : 'happy', # same as above
    '41'    : 'cheeky',
    '28'    : 'cheeky',
    '49'    : 'cheeky', # can be used as positives
    '21'    : 'thankful',
    '24'    : 'happy',
    '47'    : 'happy',
    '8'     : 'happy',
    '16'    : 'happy',
    '63'    : 'happy',
    '23'    : 'happy',
    '59'    : 'happy',
    '61'    : 'happy',
    '18'    : 'happy',
    '60'    : 'happy'
} 

for prob in [prob]:
    # Find top emojis for each sentence. Emoji ids (0-63)
    # correspond to the mapping in emoji_overview.png
    # at the root of the torchMoji repo.
    print('Writing results to {}'.format(OUTPUT_PATH))
    scores = []
    for i, t in enumerate(TEST_SENTENCES):
        t_tokens = tokenized[i]
        t_score = [t]
        t_prob = prob[i]
        ind_top = top_elements(t_prob, 5)
        emotion_mapping = [mappings[str(i)] for i in ind_top]
        t_score.append(sum(t_prob[ind_top]))
        t_score.extend(ind_top)
        t_score.extend(emotion_mapping)
        t_score.extend([t_prob[ind] for ind in ind_top])
        scores.append(t_score)
        print(t_score)

    # with open(OUTPUT_PATH, 'w') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=str(','), lineterminator='\n')
    #     writer.writerow(['Text', 'Top5%',
    #                     'Emoji_1', 'Emoji_2', 'Emoji_3', 'Emoji_4', 'Emoji_5',
    #                     'Pct_1', 'Pct_2', 'Pct_3', 'Pct_4', 'Pct_5'])
    #     for i, row in enumerate(scores):
    #         try:
    #             writer.writerow(row)
    #         except:
    #             print("Exception at row {}!".format(i))
