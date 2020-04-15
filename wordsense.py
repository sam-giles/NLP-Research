from nltk.corpus import wordnet
import time
import os

#GETS EACH LEMMA AND FILTERS OUT COMPOUND WORDS
def generateLemmas(synset):
    lemmas = []
    for l in synset.lemmas():
        #filter out compound words
        if '-' not in l.name() and '_' not in l.name():
            lemmas.append(l.name()) 
    lemmas = list(set(lemmas))
    return lemmas

#returns a dictionary of distance lists
def generateDistanceLists(word, distance):
    distanceLists = {}
    syns = wordnet.synsets(word)
    #handle distance of 0 manually
    distanceLists[0] = syns.copy()
    visitedSynsets = set(syns.copy())
    if distance is not 0:
        for i in range(1, distance+1):
            print('Distance ' + str(i))
            distanceLists[i] = []
            distanceList = []
            #get all synsets distance one away from each starting synset
            for syn in syns:
                hyps = syn.hypernyms()
                hypos = syn.hyponyms()
                distanceList.extend(hyps)
                distanceList.extend(hypos)
            #set those syns as the starting point for the next iteration
            syns = distanceList.copy()
            for item in distanceList:
                if item not in visitedSynsets:
                    distanceLists[i].append(item)
                    visitedSynsets.add(item)
    
    return distanceLists

#writes the final results to a file, printing each lemma for each synset generated
def writeDistanceListsToFile(distanceLists, filename):
    #create a new directory for the word
    os.mkdir(filename)
    pathname = os.path.join(filename, filename + '_distance_lists.txt')
    f = open(pathname, "w")
    for i in range(0,len(distanceLists)):
        f.write(str(i) + ' ')
        for syn in distanceLists[i]:
            for lemma in generateLemmas(syn):
                f.write(str(lemma) + ' ')
        f.write('\n')
    f.close()


############
##MAIN
############
def main(term='chemistry', val=6):
    start_time = time.time()

    #define the word we use
    #departure
    #chemistry
    word = term

    #n is the distance away from the word
    n = val

    print("Generating distance lists for " + word + "...")
    results = generateDistanceLists(word, n)
    print("Finished generating distance lists after", time.time()-start_time, "seconds.")

    #write the lists to a file
    print('Saving distance lists to a file...')
    writeDistanceListsToFile(results, word)
    print("Finished saving distance lists to file after", time.time()-start_time, "seconds.")

    print("Total execution time:", time.time()-start_time)

if __name__ == '__main__':
    main()
