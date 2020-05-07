import sentences
import nltk
from gensim.models import Word2Vec
import re
import time
from nltk.corpus import wordnet as wn
import string
from nltk.corpus import stopwords

def preprocessCorpus(corpus):
    #remove stopwords
    stop_words = set(stopwords.words('english'))
    corpus = [[word for word in sentence if word not in stop_words] for sentence in corpus]

    #lowercase
    corpus = [[word.lower() for word in sentence] for sentence in corpus]

    #remove punctuation
    corpus = [[word.translate(str.maketrans('', '', string.punctuation)) for word in sentence] for sentence in corpus]

    return corpus

start_time = time.time()

#read in the sentences that we already disambiguated
sentences_list = sentences.sentences
d_sentences_list = sentences.d_sentences

#change into the right format
sentences_list = [re.sub("[^\w]", " ",  sentence).split() for sentence in sentences_list]
d_sentences_list = [re.sub("[^\w]", " ",  sentence).split() for sentence in d_sentences_list]

#preprocess the sentences
sentences_list = preprocessCorpus(sentences_list)
d_sentences_list = preprocessCorpus(d_sentences_list)

#use gutenberg corpus
corpus = nltk.corpus.gutenberg.sents()
#do preprocessing
corpus = preprocessCorpus(corpus)
d_corpus = corpus.copy()

#only use the first 20% of each corpus for speed
# corpus = corpus[:int(len(corpus)/5)]
# d_corpus = d_corpus[:int(len(d_corpus)/5)]

#add sentences
corpus.extend(sentences_list)
d_corpus.extend(d_sentences_list)

#the model takes a list of documents which are a list of sentences
#train the model
model = Word2Vec(corpus)
d_model = Word2Vec(d_corpus)

#list of synsets for departure
dl = wn.synsets('departure')
#the disambiguated words that match each synset
wl = ['departure1', 'departure2', 'departure3']

#print results
for i in range(len(dl)):
    for j in range(len(wl)):
        print(dl[i], dl[j], 'Shortest path: ', dl[i].shortest_path_distance(dl[j]), 'Word2Vec distance: ', d_model.wv.distance(wl[i], wl[j]))

# for i in range(len(dl)):
#     for j in range(len(wl)):
#         print(dl[i], dl[j], 'Shortest path: ', dl[i].shortest_path_distance(dl[j]), 'Word2Vec distance: ', model.wv.distance(wl[i], wl[j]))

print("Total execution time:", time.time()-start_time)
