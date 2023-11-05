import pandas as pd

dataset = pd.read_csv("../data/interim/separated_tox.csv")
dataset = dataset.set_index(dataset.columns[0])
dataset.index.name = "Index"
dataset.head()

import os
import sys

def add_sys_path(p):
    p = os.path.abspath(p)
    print(p)
    if p not in sys.path:
        sys.path.append(p)

add_sys_path('../detox/emnlp2021/style_transfer/condBERT')

from importlib import reload
import condbert
reload(condbert)
from condbert import CondBertRewriter
import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import pickle
from tqdm.auto import tqdm, trange

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
model.to(device);

vocab_root = '../detox/emnlp2021/style_transfer/condBERT/vocab/'

with open(vocab_root + "negative-words.txt", "r") as f:
    s = f.readlines()
negative_words = list(map(lambda x: x[:-1], s))
with open(vocab_root + "toxic_words.txt", "r") as f:
    ss = f.readlines()
negative_words += list(map(lambda x: x[:-1], ss))

with open(vocab_root + "positive-words.txt", "r") as f:
    s = f.readlines()
positive_words = list(map(lambda x: x[:-1], s))

import pickle
with open(vocab_root + 'word2coef.pkl', 'rb') as f:
    word2coef = pickle.load(f)

token_toxicities = []
with open(vocab_root + 'token_toxicities.txt', 'r') as f:
    for line in f.readlines():
        token_toxicities.append(float(line))
token_toxicities = np.array(token_toxicities)
token_toxicities = np.maximum(0, np.log(1/(1/token_toxicities-1)))   # log odds ratio

# discourage meaningless tokens
for tok in ['.', ',', '-']:
    token_toxicities[tokenizer.encode(tok)][1] = 3

for tok in ['you']:
    token_toxicities[tokenizer.encode(tok)][1] = 0

reload(condbert)
from condbert import CondBertRewriter

editor_1 = CondBertRewriter(
    model=model,
    tokenizer=tokenizer,
    device=device,
    neg_words=negative_words,
    pos_words=positive_words,
    word2coef=word2coef,
    token_toxicities=token_toxicities,
)

original_sentences = list(test_dataset['toxic'])
translated_sentences = []

for i, line in enumerate(tqdm(original_sentences)):
    inp = line.strip()
    out = editor_1.translate(inp, prnt=False).strip()
    translated_sentences.append(out)

with open('results1.txt', 'w') as file:
    for item in translated_sentences:
        file.write("%s\n" % item)

corpus_tox = [' '.join([w if w in vocab else '<unk>' for w in line.strip().split()]) for line in dataset['toxic']]
corpus_norm = [' '.join([w if w in vocab else '<unk>' for w in line.strip().split()]) for line in dataset['non-toxic']]

threshold = 4

sc = NgramSalienceCalculator(corpus_tox, corpus_norm, False)
seen_grams = set()

with open(neg_out_name, 'a') as neg_out, open(pos_out_name, 'a') as pos_out:
    for gram in set(sc.tox_vocab.keys()).union(set(sc.norm_vocab.keys())):
        if gram not in seen_grams:
            seen_grams.add(gram)
            toxic_salience = sc.salience(gram, attribute='tox')
            polite_salience = sc.salience(gram, attribute='norm')
            if toxic_salience > threshold:
                neg_out.writelines(f'{gram}\n')
            elif polite_salience > threshold:
                pos_out.writelines(f'{gram}\n')

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=10000))

X_train = corpus_tox + corpus_norm
y_train = [1] * len(corpus_tox) + [0] * len(corpus_norm)
pipe.fit(X_train, y_train)

word2coef = {w: coefs[idx] for w, idx in pipe[0].vocabulary_.items()}

import pickle
with open(vocab_root + '/word2coef_2.pkl', 'wb') as f:
    pickle.dump(word2coef, f)

from collections import defaultdict
toxic_counter = defaultdict(lambda: 1)
nontoxic_counter = defaultdict(lambda: 1)

for text in tqdm(corpus_tox):
    for token in tokenizer.encode(text):
        toxic_counter[token] += 1
for text in tqdm(corpus_norm):
    for token in tokenizer.encode(text):
        nontoxic_counter[token] += 1

token_toxicities = [toxic_counter[i] / (nontoxic_counter[i] + toxic_counter[i]) for i in range(len(tokenizer.vocab))]

with open(vocab_root + '/token_toxicities_2.txt', 'w') as f:
    for t in token_toxicities:
        f.write(str(t))
        f.write('\n')

with open(vocab_root + "/negative-words.txt", "r") as f:
    s = f.readlines()
negative_words = list(map(lambda x: x[:-1], s))

with open(vocab_root + "/positive-words.txt", "r") as f:
    s = f.readlines()
positive_words = list(map(lambda x: x[:-1], s))

import pickle
with open(vocab_root + '/word2coef_2.pkl', 'rb') as f:
    word2coef = pickle.load(f)

token_toxicities = []
with open(vocab_root + '/token_toxicities_2.txt', 'r') as f:
    for line in f.readlines():
        token_toxicities.append(float(line))
token_toxicities = np.array(token_toxicities)
token_toxicities = np.maximum(0, np.log(1/(1/token_toxicities-1)))   # log odds ratio

# discourage meaningless tokens
for tok in ['.', ',', '-']:
    token_toxicities[tokenizer.encode(tok)][1] = 3

for tok in ['you']:
    token_toxicities[tokenizer.encode(tok)][1] = 0

def adjust_logits(logits, label=0):
    return logits - token_toxicities * 100 * (1 - 2 * label)

predictor = MaskedTokenPredictorBert(model, tokenizer, max_len=250, device=device, label=0, contrast_penalty=0.0, logits_postprocessor=adjust_logits)

editor = CondBertRewriter(
    model=model,
    tokenizer=tokenizer,
    device=device,
    neg_words=negative_words,
    pos_words=positive_words,
    word2coef=word2coef,
    token_toxicities=token_toxicities,
    predictor=predictor,
)

chooser = EmbeddingSimilarityChooser(sim_coef=10, tokenizer=tokenizer)

translated_sentences_2 = []

for i, line in enumerate(tqdm(original_sentences)):
    inp = line.strip()
    out = editor.translate(inp, prnt=False)
    translated_sentences_2.append(out)

with open('results3.txt', 'w') as file:
    for item in translated_sentences_2:
        file.write("%s\n" % item)