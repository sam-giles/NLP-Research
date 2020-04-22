import sentences
import nltk

sentences_list = sentences.sentences
d_sentences_list = sentences.d_sentences
splitSentences = [sentence.split() for sentence in sentences_list]
sentences_list = [word for sentence in splitSentences for word in sentence] 

corpus = list(nltk.corpus.gutenberg.words('austen-emma.txt'))
corpus_list = [word for word in corpus]
print(len(corpus_list))
corpus_list.extend(sentences_list)
print(len(corpus_list))