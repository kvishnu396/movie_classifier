import os
from pickle import dump
from nltk.tokenize import word_tokenize as tokenize
from phrases import phrases

pos_dir = 'dataset/train/pos/'
neg_dir = 'dataset/train/neg/'

pos_files = [(1,x) for x in os.listdir(pos_dir)][:500]
neg_files = [(0,x) for x in os.listdir(neg_dir)][:500]
train_files = pos_files + neg_files
phrase_set = set()
phrase_list = [None]
freq_list = [None]
n = 0

for label, tfile in train_files:
    if label is 1:
        fdir = pos_dir
    else:
        fdir = neg_dir
    with open(fdir+tfile) as f:
        text = f.read()
        tokens = tokenize(text)
        phrasel = phrases(tokens)
        sp = set(phrasel)
    new_phrases = list(sp - phrase_set)
    phrase_list = phrase_list + new_phrases
    freq_list = freq_list + [0]*len(new_phrases)
    phrase_set = phrase_set or sp
    for p in phrasel:
        i = phrase_list.index(p)
        freq_list[i] += 1
    n += 1
    print(n)

print('done')

with open('phrase.set', 'wb') as f:
    dump(phrase_list,f)
