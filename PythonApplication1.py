import itertools
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import json

#-------------------------------------------------------------GLOBAL VARIABLES---------------------------------------------------------------------

#these are for evaluating the performance of my scalability solution

numberOfDuplicatesFound = 0
totalNumberOfDuplicates = 0
numberOfComparisons = 0

#--------------------------------------------------------------------------------------------------------------------------------------------------


def readData():
    x = open('TVs-all-merged.json')
    data = json.load(x)
    inputData = []
    for item in data:
        #iau fiecare element din fisier si ma plimb prin featuresMap
        #featuresMap este un dictionar de tip {cheie:valoare}
        #iau fiecare valoare si o pun intr-o lista
        #la final in lista mea inputData voi avea un sir alcatuit din toate cheile din featuresMap separate prin spatiu
        features = data[item][0]["featuresMap"]
        values = []
        for key in features:
            values.append(features[key])
        inputData.append(' '.join(values))
    return inputData


#-------------------------------------------------------------K-SHINGLING---------------------------------------------------------------------

#se plimba prin toate elemnetele din inputData si le sparg in subsiruri de cate k elemente 
def createShingle(inputData, k):
    shingles = []
    for inputRow in inputData:
        shingle = []
        for i in range(len(inputRow) - k + 1):
            shingle.append(inputRow[i:i+k])
        #set elimina duplicatele
        shingles.append(set(shingle))
    return shingles

#in vocabular voi avea toate shingle-urile 
def createVocabulary(shingles):
    vocabulary = shingles[0]
    for i in range(1, len(shingles)):
        vocabulary = vocabulary.union(shingles[i])
    return set(vocabulary)


def createBinaryVectors(vocabulary, shingles):
    binaryVectors = []
    for shingle in shingles:
        #un astfel de vector va avea aceeasi lungime cu shingle
        #se plimba prin toate elementele dintr-un shingle si daca acel element se afla si in vocabulary atunci pune 1 pe pozitia corespunzatoare in vector; altfel pune 0 
        vector = [ 1 if x in shingle else 0 for x in vocabulary]
        binaryVectors.append(vector)
    return binaryVectors


#-------------------------------------------------------------MINHASHING---------------------------------------------------------------------

#o apelez in functia buildMinhashFunc - dedesupt
def createHashVector(size):
    #imi creeaza o lista care are elementele de la 1 la size - in acest caz size va fi lungimea vocabularului
    hashList = list(range(1,size))

    #imi schimba random ordinea elementelor
    shuffle(hashList)
    return hashList


#noVectors = cati vectori hash imi va crea - este un numar random - cu cat este mai mare cu atat este o acuratete mai crescuta
#vocabularySize = cate elemente are vocabularul
def buildMinhashVectors(vocabularySize, noVectors):
    hashes = []
    for _ in range(noVectors):
        hashes.append(createHashVector(vocabularySize))
    return hashes


#function to create signatures = procesul de a converti binaryVectors in dense vectors
def createSignatures(minhashVectors, vocabulary, binaryVectors):
    signatures = []
    for binaryVector in binaryVectors:
        signature = []
        for minhashVector in minhashVectors:
            for i in range(1, len(vocabulary)+1):
                #index = imi spune ca elementul i se afla pe pozitia index in minhashVector
                index = minhashVector.index(i)
                signatureVal = binaryVector[index]
                if signatureVal == 1:
                    signature.append(index)
                    break
        signatures.append(signature)
    return signatures


#-------------------------------------------------------------BAND AND HASH---------------------------------------------------------------------

def smallestDivisor(signatures):
    lengthOfSginatureVectors = []
    for signature in signatures:
        lengthOfSginatureVectors.append(len(signature))
    #in cel mai rau caz cmmdc va fi lungimea minima a unui vector signature

    cmmdc = 1
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

#imi imparte fiecare signature in subvectori de lungime b - trebuie sa ma asigur ca len(signature) % b == 0 => trebuie sa aleg cmmdc(signatures[i]) unde i=0, len(signatures)
def createBands(signatures, b):
    bands = []
    for signature in signatures:
        r = int(len(signature) / b)
        subvecs = []
        for i in range(0, len(signature), r):
            subvecs.append(signature[i: i+r])
        bands.append(subvecs)
    return bands

def findCandidatePairs(bands):
    global numberOfComparisons
   
    canditatePairs = []

    for i in range (0, len(bands)):
        for j in range(i+1, len(bands)):
            for i_rows, j_rows in zip(bands[i], bands[j]):
                if i_rows == j_rows:
                    numberOfComparisons += 1
                    #inseamna ca televizoarele i si j vor fi considerati candidate pair
                    canditatePairs.append((i, j))
                    break

    
    return canditatePairs

#-------------------------------------------------------------EVALUATE PERFORMANCE---------------------------------------------------------------------


def evaluatePerformance():

    pairQuality = numberOfDuplicatesFound/numberOfComparisons
    pairCompleteness = numberOfDuplicatesFound/totalNumberOfDuplicates

    F1measure = 2 / ( (1/pairQuality)  + (1/pairCompleteness) )

    print("numberOfDuplicatesFound = " + str(numberOfDuplicatesFound))
    print("numberOfComparisons = " + str(numberOfComparisons))
    print("totalNumberOfDuplicates = " + str(totalNumberOfDuplicates))
    print("pair quality = " + str(pairQuality))
    print("pairCompleteness = " + str(pairCompleteness))
    print("F1 measure = " + str(F1measure))


def jaccardSimilarity(d1, d2):
    return len(d1.intersection(d2))/len(d1.union(d2))

def trueSimScores(shingles):
    pair_labels = []
    pair_sims = []
    idexes = range(len(shingles))
    for x1, x2 in itertools.combinations(zip(idexes,shingles), 2):
        pair_labels.append((x1[0], x2[0]))
        pair_sims.append(jaccardSimilarity(x1[1], x2[1]))
    return dict(zip(pair_labels, pair_sims))

def candidatePairsFunc(score_dict, threshold):
    #cu cat scorul e mai aproape de 1 cu atat perechile sunt mai similare
    return set(pair for pair, scr in score_dict.items() if scr >= threshold)


#-------------------------------------------------------------GRAPH-----------------------------------------------------------------------------

def graph(pairs, inputData, scoresDict):
    n = len(inputData)

    #vad ce scor de similitudine au perechile candidate intre ele
    pairsScoresDict = {}
    for pair in pairs:
        if scoresDict.has_key(pair):
            pairsScoresDict[pair] = scoresDict[pair]

    #** - inseamna ridicare la putere
    yval = lambda p,r,b: 1-(1-p**r)**b

    for key in pairsScoresDict:
        plt.plot(pts, yval(pairsScoresDict[key],op[0],op[1]), label=op)

    
    


#-------------------------------------------------------------MAIN FUNCTION---------------------------------------------------------------------


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

    #gasesc true pairs
    scoresDict = trueSimScores(shingles)
    truePairs = candidatePairsFunc(scoresDict, 0.8)
    totalNumberOfDuplicates = len(truePairs)
    numberOfDuplicatesFound = len(candidatePairs)
    
    #print(candidatePairs[0])
    #print("lungime vectori perechi:" + str(len(candidatePairs)))
    
    evaluatePerformance()

    
main()
