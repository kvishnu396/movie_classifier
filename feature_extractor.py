import os
import nltk
from phrases import phrases

pos_dir = 'Cooper/Sophomore/NLP/aclImdb/train/pos'
neg_dir = 'Cooper/Sophomore/NLP/aclImdb/train/neg'

pos_files = [ (1,x) for x in os.listdir(pos_dir)]
neg_files = [ (0,x) for x in os.listdir(neg_dir)]
train_files = pos_files + neg_files


