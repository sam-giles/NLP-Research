import spacy
import time
import numpy
import ast
import matplotlib.pyplot as plt 

#convert the file output back to a dictionary of lists
def readDistanceListsFromFile(filename, word):
    distanceLists = {}
    with open(word + '/' + filename,'r') as f:
        for line in f:
            tempLineList = []
            for word in line.split():
                tempLineList.append(word)
            distanceLists[int(tempLineList[0])] = list(set(tempLineList[1:]))
    f.close()
    return distanceLists   

#GETS THE VECTOR MODEL TOKENS FOR A LIST OF WORDS
def generateTokens(words, model):
    #break synonyms into a string that can be used
    syn_string = " "
    syn_string = syn_string.join(words)
    tokens = model(syn_string)
    return tokens

#GETS THE SIMILARITY BETWEEN A TOKEN AND A SET OF TOKENS
def generateSimilarity(distance, token, tokenSet):
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
        sim = token.similarity(tok)
        simList.append(sim)

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

#writes the results of the program to a file
def writeSimilaritiesToFile(similarities, filename, word):
    f = open(word + '/' + filename, "w")
    for item in similarities:
        f.write(str(item) + '\n')
    f.close()

def generateGraphFromSimilarity(word):
    file = open(word + '/' + word + '_similarities.txt', "r")
    contents = file.readlines()
    dicts = []
    for i in range(len(contents)):
        dicts.append(ast.literal_eval(contents[i]))
    distances = [item['Distance'] for item in dicts]
    means = [item['Mean'] for item in dicts]

    # x axis values 
    x = distances 
    # corresponding y axis values 
    y = means
    
    # plotting the points  
    plt.plot(x, y) 
    
    # naming the x axis 
    plt.xlabel('Distance') 
    # naming the y axis 
    plt.ylabel('Mean Similarity') 
    
    # giving a title to my graph 
    plt.title('Mean Similarity for "' + word + '" Over Distance') 
    
    # save the plot
    plt.savefig(word + '/' + word + '_graph_part1.png')
    

###########
##MAIN
###########
def main(term='chemistry'):
    #start timer
    start_time = time.time()

    #load the spacy model
    print("Loading spacy word embedding model...")
    nlp = spacy.load("en_core_web_lg")  # make sure to use larger model
    print("Finished loading model after", time.time()-start_time, "seconds.")

    #set the word to be used
    word = term

    print("Loading " + word + " distance lists from file...")
    #get the distance lists
    distanceLists = readDistanceListsFromFile(word + '_distance_lists.txt', word)
    print("Finished loading distance lists from file after", time.time()-start_time, "seconds.")

    #get the token for word
    wordToken = generateTokens([word], nlp)

    #loop to compare each distance
    print("Calculating similarities by distance...")
    similarities = []
    for i in range(0, len(distanceLists)):
        distanceTokens = generateTokens(distanceLists[i], nlp)
        similarities.append(generateSimilarity(i, wordToken, distanceTokens))
    print("Finished calculating similarities by distance after", time.time()-start_time, "seconds.")

    #print out final results to a file
    print("Saving results to file...")
    writeSimilaritiesToFile(similarities, word + '_similarities.txt', word)

    print("Total execution time:", time.time()-start_time)

if __name__ == '__main__':
    main()