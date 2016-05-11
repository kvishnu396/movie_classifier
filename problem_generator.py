import os
from pickle import load, dump
from itertools import chain
from nltk.tokenize import word_tokenize as tokenize
from scipy.sparse import lil_matrix
from phrases import phrases

pos_dir = 'dataset/train/pos/'
neg_dir = 'dataset/train/neg/'

pos_files = os.listdir(pos_dir)[:2500]
neg_files = os.listdir(neg_dir)[:2500]
train_files = chain(pos_files, neg_files)

with open('sword.set', 'rb') as f:
    sword_list = load(f)

with open('phrase.set','rb') as f:
    phrase_list = load(f)

problem = lil_matrix((5000, 17173))
n = 0

for i, tfile in enumerate(train_files):
    if i < 2500:
        fdir = pos_dir
    else:
        fdir = neg_dir
    with open(fdir+tfile) as f:
        text = f.read()
        tokens = tokenize(text)
        fphrases = phrases(tokens)
    for token in tokens:
        if token in sword_list:
            ind = sword_list.index(token)
            problem[i, ind] = 1
    for p in fphrases:
        if p in phrase_list:
            ind = phrase_list.index(p) + 3111
            problem[i, ind] = 1

with open('problem.matrix', 'wb') as f:
    dump(problem, f)
