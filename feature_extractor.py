import os
from nltk.tokenize import word_tokenize as tokenize
from phrases import phrases

pos_dir = '~/Cooper/Sophomore/NLP/aclImdb/train/pos'
neg_dir = '~/Cooper/Sophomore/NLP/aclImdb/train/neg'

pos_files = [ (1,x) for x in os.listdir(pos_dir)]
neg_files = [ (0,x) for x in os.listdir(neg_dir)]
train_files = pos_files + neg_files
phrase_set = set()
phrase_list = [None]
freq_list = [None]
n = 1

for tfile in train_files:
    with open(tfile) as f:
        text = f.read()
        tokens = tokenize(text)
        phrases = phrases(tokens)
        sp = set(phrases)
    new_phrases = list(sp - phrase_set)
    phrase_list = phrase_list + new_phrases
    freq_list = freqlist + [0]*len(new_phrases)
    phrase_set = dataset_phrases or set(phrases)
    for p in phrases:
        i = phrase_list.index(p)
        freq_list += 1
    n += 1
    print(n)

print('done')
for p, freq in zip(phrase_list, freq_list):
    with open('phrase.set', a) as f:
        f.write(p+' '+freq)
