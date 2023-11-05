import pandas as pd

dataset = pd.read_csv("../data/interim/separated_tox.csv")
dataset = dataset.set_index(dataset.columns[0])
dataset.index.name = "Index"
dataset.head()

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from nltk import ngrams
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def add_sys_path(p):
    p = os.path.abspath(p)
    print(p)
    if p not in sys.path:
        sys.path.append(p)

add_sys_path('../detox/emnlp2021/style_transfer/condBERT')

from choosers import EmbeddingSimilarityChooser
from multiword.masked_token_predictor_bert import MaskedTokenPredictorBert

class NgramSalienceCalculator():
    def __init__(self, tox_corpus, norm_corpus, use_ngrams=False):
        ngrams = (1, 3) if use_ngrams else (1, 1)
        self.vectorizer = CountVectorizer(ngram_range=ngrams)

        tox_count_matrix = self.vectorizer.fit_transform(tox_corpus)
        self.tox_vocab = self.vectorizer.vocabulary_
        self.tox_counts = np.sum(tox_count_matrix, axis=0)

        norm_count_matrix = self.vectorizer.fit_transform(norm_corpus)
        self.norm_vocab = self.vectorizer.vocabulary_
        self.norm_counts = np.sum(norm_count_matrix, axis=0)

    def salience(self, feature, attribute='tox', lmbda=0.5):
        assert attribute in ['tox', 'norm']
        if feature not in self.tox_vocab:
            tox_count = 0.0
        else:
            tox_count = self.tox_counts[0, self.tox_vocab[feature]]

        if feature not in self.norm_vocab:
            norm_count = 0.0
        else:
            norm_count = self.norm_counts[0, self.norm_vocab[feature]]

        if attribute == 'tox':
            return (tox_count + lmbda) / (norm_count + lmbda)
        else:
            return (norm_count + lmbda) / (tox_count + lmbda)
        

from collections import Counter
c = Counter()

# read words from our portion of the dataset
for fn in [dataset['toxic'], dataset['non-toxic']]:
    for line in fn:
        for tok in line.strip().split():
            c[tok] += 1

neg_out_name = "/content/detox/emnlp2021/style_transfer/condBERT/vocab/negative-words.txt"
pos_out_name = "/content/detox/emnlp2021/style_transfer/condBERT/vocab/positive-words.txt"

# read words that already are in the dictionary
with open(neg_out_name, 'r') as neg_out, open(pos_out_name, 'r') as pos_out:
    existant_pos_words = pos_out.readlines()
    for line in existant_pos_words:
        for tok in line.strip().split():
            c[tok] += 1
    existant_neg_words = neg_out.readlines()
    for line in existant_neg_words:
        for tok in line.strip().split():
            c[tok] += 1

vocab = {w for w, _ in c.most_common() if _ > 0}  # if we took words with > 1 occurences, vocabulary would be x2 smaller, but we'll survive this size