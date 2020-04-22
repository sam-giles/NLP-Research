import sentences
import nltk

sentences = sentences.sentences
splitSentences = [sentence.split() for sentence in sentences]
sentences = [word for sentence in splitSentences for word in sentence] 

corpus = nltk.corpus.gutenberg.words('bible-kjv.txt')
print(corpus)