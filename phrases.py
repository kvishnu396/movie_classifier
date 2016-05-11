#Training Program
import nltk
from nltk.collocations import *
from nltk.corpus import PlaintextCorpusReader

def filt(x):
    return x.label()=="ONE" or x.label()=="TWO" or x.label()=="THREE" or x.label()=="THREE" or x.label()=="FOUR" or x.label()=="FIVE"

#Input, a tokenized text
#Output, a list of phrases
#Each element in the list is a tuple of tuples
#An element is in the form of
#(("word1", POS), ("word2", POS))
def phrases(text):
    pos_bigram_measures = nltk.collocations.BigramAssocMeasures()
    pos_finder = BigramCollocationFinder.from_words(text)
    pos = nltk.pos_tag(text)

    grammar = r"""
    ONE:
        {<JJ><NN|NNS><.*>}
    TWO:
        {<RB|RBR|RBS><JJ><^NN.>}
    THREE:
        {<JJ><JJ><[^NN].><.*>}
    FOUR:
        {<NN|NNS><JJ><^NN.><.*>}
    FIVE:
        {<RB|RBR|RBS><VB|VBD|VBN|VBG><.*>}
    """
    cp = nltk.RegexpParser(grammar)
    output = cp.parse(pos)

    phrases = []
    for subtree in output.subtrees(filter = filt): # Generate all subtrees
        phrases.append(subtree.productions())

    phraseList = []
    for prod in phrases:
        phraseList.append(prod[0].rhs()[:2])
    return phraseList
