import os
from pickle import load, dump
from itertools import chain
from nltk.tokenize import word_tokenize as tokenize
from scipy.sparse import lil_matrix
from phrases import phrases

pos_dir = '/home/ihsan/Cooper/Sophomore/NLP/aclImdb/train/pos/'
neg_dir = '/home/ihsan/Cooper/Sophomore/NLP/aclImdb/train/neg/'

#pos_files = [(1,x) for x in os.listdir(pos_dir)][:5]
#neg_files = [(0,x) for x in os.listdir(neg_dir)][:5]
pos_files = os.listdir(pos_dir)[:5]
neg_files = os.listdir(neg_dir)[:5]
train_files = chain(pos_files, neg_files)

with open('sword.set', 'rb') as f:
    sword_list = load(f)

with open('phrase.set','rb') as f:
    phrase_list = load(f)

problem = lil_matrix((1000,18345))
n = 0

for i, tfile in enumerate(train_files):
    if i < 5:
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
            ind = phrase_list.index(p) + 4283
            problem[i, ind] = 1

with open('problem.matrix', 'wb') as f:
    dump(problem, f)
