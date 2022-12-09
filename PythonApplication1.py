import itertools
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import json

# -------------------------------------------------------------GLOBAL VARIABLES---------------------------------------------------------------------

# these are for evaluating the performance of my scalability solution

numberOfDuplicatesFound = 0
totalNumberOfDuplicates = 0
numberOfComparisons = 0


# --------------------------------------------------------------------------------------------------------------------------------------------------


def readData():
    x = open('TVs-all-merged.json')
    data = json.load(x)
    inputData = []
    for item in data:
        # I take every element from the file and I walk through featuresMap
        # featuresMap is a dictionary of type {key:value}
        # I take every value and I put it in a list
        # in the end, in my list named inputDaya I will have a string made with all the keys from featuresMap separated by space
        features = data[item][0]["featuresMap"]
        values = []
        for key in features:
            values.append(features[key])
        inputData.append(' '.join(values))
    return inputData


# -------------------------------------------------------------K-SHINGLING---------------------------------------------------------------------

# is walking through all the elements from inputData and it breaks them in subsets of k elements
def createShingle(inputData, k):
    shingles = []
    for inputRow in inputData:
        shingle = []
        for i in range(len(inputRow) - k + 1):
            shingle.append(inputRow[i:i + k])
        # set removes the duplicates
        shingles.append(set(shingle))
        #print(f"A number of {len(shingles)} unique shingles have been found, out of {len(inputData)} possible.")
    return shingles

# in vocabulary I will have all the shingles
def createVocabulary(shingles):
    vocabulary = shingles[0]
    for i in range(1, len(shingles)):
        vocabulary = vocabulary.union(shingles[i])
    return set(vocabulary)


def createBinaryVectors(vocabulary, shingles):
    binaryVectors = []
    for shingle in shingles:
        # this kind of vector will have the same lenght as a shingle
        # is walking through all the elements from a shingle and if that element is also in the vocabulary, then it will generate 1 in the coresponding position in the vector; otherwise it will generate 0
        vector = [1 if x in shingle else 0 for x in vocabulary]
        binaryVectors.append(vector)
    return binaryVectors


# -------------------------------------------------------------MINHASHING---------------------------------------------------------------------

# I call it in the function buildMinhashFunc - under
def createHashVector(size):
    # it will create a list with elements from 1 to size - in this case it will be the lengjt of the vocabulary
    hashList = list(range(1, size))

    # it randomly changes the orderder of the elements
    shuffle(hashList)
    return hashList


# noVectors =  how many vectors hash will create - is a random number - the bigger the number the higher the accuracy
# vocabularySize = how many elements are in the vocabulary
def buildMinhashVectors(vocabularySize, noVectors):
    hashes = []
    for _ in range(noVectors):
        hashes.append(createHashVector(vocabularySize))
    return hashes


# function to create signatures = the process of converting binaryVectors into dense vectors
def createSignatures(minhashVectors, vocabulary, binaryVectors):
    signatures = []
    for binaryVector in binaryVectors:
        signature = []
        for minhashVector in minhashVectors:
            for i in range(1, len(vocabulary) + 1):
                # index = is telling me that the element i is in the index position in minhashVector
                index = minhashVector.index(i)
                signatureVal = binaryVector[index]
                if signatureVal == 1:
                    signature.append(index)
                    break
        signatures.append(signature)
    return signatures


# -------------------------------------------------------------BAND AND HASH---------------------------------------------------------------------

def smallestDivisor(signatures):
    lengthOfSginatureVectors = []
    for signature in signatures:
        lengthOfSginatureVectors.append(len(signature))
    # in cel mai rau caz cmmdc va fi lungimea minima a unui vector signature

    cmmdc = 1 #cmmdc = the lowest common denominator
    ok = True
    for i in range(2, min(lengthOfSginatureVectors)):
        for length in lengthOfSginatureVectors:
            if length % i != 0:
                ok = False
                break
        if ok == True:
            cmmdc = i
            break
    return cmmdc


# it breaks every signature in subvectors of lenght b - I have to make sure that len(signature) % b == 0 => I must choose cmmdc(signatures[i]), where i=0, len(signatures)
def createBands(signatures, b):
    bands = []
    for signature in signatures:
        r = int(len(signature) / b)
        subvecs = []
        for i in range(0, len(signature), r):
            subvecs.append(signature[i: i + r])
        bands.append(subvecs)
    return bands


def findCandidatePairs(bands):
    global numberOfComparisons

    canditatePairs = []

    for i in range(0, len(bands)):
        for j in range(i + 1, len(bands)):
            for i_rows, j_rows in zip(bands[i], bands[j]):
                if i_rows == j_rows:
                    numberOfComparisons += 1
                    # this means that televisions i and j will be considered candidate pairs
                    canditatePairs.append((i, j))
                    break

    return canditatePairs


# -------------------------------------------------------------EVALUATE PERFORMANCE---------------------------------------------------------------------


def evaluatePerformance():
    pairQuality = numberOfDuplicatesFound / numberOfComparisons
    pairCompleteness = numberOfDuplicatesFound / totalNumberOfDuplicates

    F1measure = 2 / ((1 / pairQuality) + (1 / pairCompleteness))

    print("numberOfDuplicatesFound = " + str(numberOfDuplicatesFound))
    print("numberOfComparisons = " + str(numberOfComparisons))
    print("totalNumberOfDuplicates = " + str(totalNumberOfDuplicates))
    print("pair quality = " + str(pairQuality))
    print("pairCompleteness = " + str(pairCompleteness))
    print("F1 measure = " + str(F1measure))

# --------------------------------------------------------JACCARD SIMILARITY-------------------------------------------------------------
def jaccardSimilarity(d1, d2):
    return len(d1.intersection(d2)) / len(d1.union(d2))


def trueSimScores(shingles):
    pair_labels = []
    pair_sims = []
    idexes = range(len(shingles))
    for x1, x2 in itertools.combinations(zip(idexes, shingles), 2):
        pair_labels.append((x1[0], x2[0]))
        pair_sims.append(jaccardSimilarity(x1[1], x2[1]))
        #for pair, score in zip(pair_labels, pair_sims):
            #print(f"{pair_labels}\t{score: .3f}")
    return dict(zip(pair_labels, pair_sims))


def candidatePairsFunc(score_dict, threshold):
    # the closer the score is to 1 the more similar are the pairs
    return set(pair for pair, scr in score_dict.items() if scr >= threshold)

#-------------------------------------------------------------CLUSTERING---------------------------------------------------------------------

#def clustering(candidatePairs):
    #linkage_data = linkage(data, method='ward', metric='euclidean')
    #dendrogram(linkage_data)
    #hierarchical_cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    #labels = hierarchical_cluster.fit_predict(candidatePairs)
    #x = []
    #y = []
    #for item in candidatePairs:
        #x.append(item[0])
        #y.append(item[1])
    #plt.scatter(x, y, c=labels)
    #plt.show()


def evaluatePerformance():

    pairQuality = numberOfDuplicatesFound/numberOfComparisons
    pairCompleteness = numberOfDuplicatesFound/totalNumberOfDuplicates

    F1measure = 2 / ( (1/pairQuality)  + (1/pairCompleteness) )

    print("F1 measure = " + str(F1measure))

# -------------------------------------------------------------MAIN FUNCTION---------------------------------------------------------------------


def main():
    global totalNumberOfDuplicates
    global numberOfDuplicatesFound

    k = 3
    inputData = readData()
    shingles = createShingle(inputData, k)
    vocabulary = createVocabulary(shingles)
    binaryVectors = createBinaryVectors(vocabulary, shingles)
    minhashVectors = buildMinhashVectors(len(vocabulary), 20)
    signatures = createSignatures(minhashVectors, vocabulary, binaryVectors)
    bands = createBands(signatures, smallestDivisor(signatures))
    candidatePairs = findCandidatePairs(bands)

    # I find the true pairs
    truePairs = candidatePairsFunc(trueSimScores(shingles), 0.8)
    totalNumberOfDuplicates = len(truePairs)
    numberOfDuplicatesFound = len(candidatePairs)

    evaluatePerformance()


main()

