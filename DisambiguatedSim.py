import sentences
import nltk
from gensim.models import Word2Vec
import re
import time
from nltk.corpus import wordnet as wn

start_time = time.time()

t = wn.synsets('departure')[0]
s = wn.synsets('chemistry')[0]
print(t, s)
print(t.shortest_path_distance(s))
print(t.wup_similarity(s))

#read in the sentences that we already disambiguated
sentences_list = sentences.sentences
d_sentences_list = sentences.d_sentences

#change into the right format
sentences_list = [re.sub("[^\w]", " ",  sentence).split() for sentence in sentences_list]
d_sentences_list = [re.sub("[^\w]", " ",  sentence).split() for sentence in d_sentences_list]

#use gutenberg corpus
corpus = nltk.corpus.gutenberg.sents()
#convert into type 'list'
corpus = [item for item in corpus]
d_corpus = corpus.copy()
#only use the first 20% of each corpus for speed
corpus = corpus[:int(len(corpus)/5)]
d_corpus = d_corpus[:int(len(d_corpus)/5)]
#add sentences
corpus.extend(sentences_list)
d_corpus.extend(d_sentences_list)

# #the model takes a list of documents which are a list of sentences
model = Word2Vec(corpus)
# #d_model = model = Word2Vec(d_corpus_list, min_count=1,size= len(d_corpus_list),workers=3, window =3, sg = 1)

# for sentence in corpus:
#     for word in sentence:
#         try:
#             print('departure', word, model.wv.similarity('departure', word))
#         except:
#             print("Word " + word + " not in vocabulary")

print("Total execution time:", time.time()-start_time)
