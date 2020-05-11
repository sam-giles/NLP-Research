import sentences
import nltk
from gensim.models import Word2Vec
import re
import time
from nltk.corpus import wordnet as wn
import string
from nltk.corpus import stopwords
import similarity
import numpy

def preprocessCorpus(corpus):
    #remove stopwords
    stop_words = set(stopwords.words('english'))
    corpus = [[word for word in sentence if word not in stop_words] for sentence in corpus]

    #lowercase
    corpus = [[word.lower() for word in sentence] for sentence in corpus]

    #remove punctuation
    corpus = [[word.translate(str.maketrans('', '', string.punctuation)) for word in sentence] for sentence in corpus]

    return corpus

def generateDistance(distance, token, tokenSet, model):
    similarities = {}
    similarities['Distance'] = distance
    similarities['Set Size'] = len(tokenSet)
    if len(tokenSet) == 0:
        similarities['Mean'] = None
        similarities['Min'] = None
        similarities['Max'] = None
        similarities['Standard Deviation'] = None
        return similarities
    #compare each word to each word in the other synset
    simList = []
    for tok in tokenSet:
        try:
            sim = model.wv.distance(token, tok)
            simList.append(sim)
        except:
            pass

    vals = numpy.array(simList)
    avg = numpy.mean(vals)
    minimum = numpy.amin(vals)
    maximum = numpy.amax(vals)
    stdev = numpy.std(vals)

    similarities['Mean'] = avg
    similarities['Min'] = minimum
    similarities['Max'] = maximum
    similarities['Standard Deviation'] = stdev

    return similarities

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
corpus = corpus[:int(len(corpus)/5)]
d_corpus = d_corpus[:int(len(d_corpus)/5)]

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
        print(dl[i], dl[j], 'Shortest path:', dl[i].shortest_path_distance(dl[j]), 'Word2Vec distance:', d_model.wv.distance(wl[i], wl[j]))

distanceLists = similarity.readDistanceListsFromFile('departure_distance_lists.txt', 'departure')

dis4 = generateDistance(4, 'departure', distanceLists[4], model)
dis6 = generateDistance(6, 'departure', distanceLists[6], model)

print(dis4)
print(dis6)

print("Total execution time:", time.time()-start_time)
