import nltk
import unicodedata
from pickle import dump
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn

#sort out words that are subjective or have a connection to both good and bad
vocab = open('aclImdb/imdb.vocab')
words = vocab.readlines()

subject_words = []
forms = ['.a.01','.a.02']

#initiliaze subject_words with subjective words from vocab list
for w in words:
	lemmatizer = WordNetLemmatizer()
	try:
		root = lemmatizer.lemmatize(w)
	except:
		pass
	if isinstance(root, unicode):
		root = unicodedata.normalize('NFKD', root).encode('ascii','ignore')
	root = root.strip()	
	for f in forms:
		j = root + f
		try:
			if (swn.senti_synset(j).obj_score() < 0.5):
				subject_words.append(root)
				break
			else:
				pass
		except:
			pass

out = open('sword.set', 'w')
dump(subject_words, out)
out.close()
print len(subject_words)
 
