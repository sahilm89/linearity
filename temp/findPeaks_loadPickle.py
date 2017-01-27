import pickle
import numpy as np

def unpickleObject(name ):
    ''' Load pickled object '''
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def calculateReliability(expectedNumer, actualNumber):
    reliability = np.exp( - (1 - ( actualNumber / float(expectedNumber) ) ) ** 2 )
    return reliability 

def findLongestIncrementingSubSequence (sequenceDict, retrieveSorted = 0):
    # getting ordered list for the dictionary with increasing indices:
    sortedItems = map(lambda x:x[1], sorted(sequenceDict.items(), key=lambda x:x[0]))
    ##############################################################################
    globalLiveSequences = {} # (value, startIndex): length
    currentLiveSequences = {} # value: length
    
    for i in range(len(sortedItems)):
        updatedLiveSequences = {} # updated value: length
        for item in sortedItems[i]:
            if (item - 1) in currentLiveSequences:
                updatedLiveSequences[item] = currentLiveSequences[item-1] + 1
            else:
                updatedLiveSequences[item] = 1

        deadSet = set(currentLiveSequences.keys()) - set(updatedLiveSequences.keys() + [key - 1 for key in updatedLiveSequences.keys()])
        for item in deadSet:
            globalLiveSequences[ (item-currentLiveSequences[item]+1, i-currentLiveSequences[item]) ] = currentLiveSequences[item]
    
        currentLiveSequences = updatedLiveSequences

    if (retrieveSorted):
        retrievedList = sorted(globalLiveSequences.items(), key=lambda x:x[1], reverse=True)[:retrieveSorted] 
    else:
        retrievedList = sorted(globalLiveSequences.items(), key=lambda x:x[1], reverse=True)
 
    return retrievedList, globalLiveSequences

peakDict = unpickleObject('indexedWholeTrace_RS45')
peakReliability = {}
expectedNumber = 4
thresholdReliability = 5e-2 

for key in peakDict.keys():
    if len(peakDict[key]):
        peakReliability.update({ key: calculateReliability(expectedNumber, len(peakDict[key])) })
    else:
        peakReliability.update({ key: 0. })

for key in peakReliability.keys():
    if peakReliability[key] < thresholdReliability:
        peakDict[key] = []

seqList = []

for value in peakDict.values():
    seqList.extend(value)#.flatten()

minVal, maxVal =  np.min(seqList), np.max(seqList)

sequenceList = {}

for key in range(minVal, maxVal):
    sequenceList.update ({key: []})

for key in peakDict.keys():
    if len(peakDict[key]):
        for index in peakDict[key]:
            if (index in sequenceList):
                sequenceList[index].append( key )

sequenceDict= sequenceList.copy()
retrievedList, globalLiveSequences = findLongestIncrementingSubSequence (sequenceDict, retrieveSorted = 5)
print retrievedList

