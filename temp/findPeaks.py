#!/usr/bin/python
'''
This algorithm is to find peaks in the photodiode trace and assign them coordinate values.
The inputs for the algorithm are:
    1. Coordinate files.
    2. Photodiode template trace to train the classifier.
    3. Photodiode trace from the random stimulation dataset for testing and classifying.

The algorithm will do the following:
    1. Train a classifier on the template sequence by assigning the peaks of the trace to a coordinate set.
    2. Encode these coord sets as letters. ( Do you have to?)
    3. Identify the blank in the sequence.( Zone with high information content).
    4. Chunk the data into fragments like the above dataset.
    5. Run a loop over the random stimulation dataset, and starting from the blank, over the chunks.
    6. Parallely run over the template chunks, but in steps of the chunk size ( as opposed to the moving window).
    7. Keep going until a mismatch is found, then move in single steps until a match is found, and mark the position as break.
    8. Again keep going and repeat.
    9. Score longer and smaller fragments.
'''

import numpy as np
import util
import scipy.signal as ss
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import pickle
import itertools as it
import sys
import random

def detectWavelets(Trace,peakWidth, pattern, noise_perc):
    ''' Detects peaks in a given 1 D trace, with given parameters '''
    indices = ss.find_peaks_cwt(Trace,np.arange(peakWidth[0],peakWidth[1]), wavelet= pattern, noise_perc=noise_perc)
    return indices

def detectPeaksAndValleys(Trace):
    ''' Detects valleys and peaks in a given 1 D trace, with given parameters '''
    # for local maxima
    maxIndices = argrelextrema(Trace, np.greater)

    # for local minima
    minIndices= argrelextrema(Trace, np.less)
    return maxIndices, minIndices

def filterByExtremas(Trace):
    ''' Extract maxima and minima from a trace '''
    maxIndices, minIndices = detectPeaksAndValleys(Trace)
    totalIndices = np.append(maxIndices, minIndices)
    filteredSignal = Trace[np.sort(totalIndices)]
    return filteredSignal, maxIndices, minIndices

def locateBlank(blankTemplate, Trace):
    ''' Locates the index of the blank '''
    corr, pvalue = correlateAgainstTrace(blankTemplate, Trace)
    blankIndices = findPeakIndexFromCorrelation(corr, pValue = pvalue,correlation_tolerance = 0.90, pValue_tolerance = 0.005)
    return blankIndices 

def chunkTrace( trace, chunkSize = [], numChunks = []):
    ''' Chunks the trace into trace/chunkSize fragments '''
    if chunkSize:
        numChunks = len(trace)/chunkSize 
        residual = len(trace)%chunkSize
    elif numChunks:
        chunkSize = len(trace)/numChunks
        residual = len(trace)%numChunks
    else:
        print "You need either number of chunks or chunk size"
        sys.exit()

    if not (residual):
        chunks = np.split(trace,numChunks)
        anomaly = 0
    else:
        #increaseIndices = random.sample(range(numChunks),residual) # Picking remainder number of indices
        increaseIndices = [ i for i in range(numChunks) if i%3 !=0 ] # Picking remainder number of indices
        #print increaseIndices
        chunkIndices = []
        totalIncrement = 0
        for i in range(numChunks-1): ## One less break to make a given number of chunks
            if i in increaseIndices:
                increment = chunkSize + 1 ## Incrementing in excess of 1
            else:
                increment = chunkSize
            totalIncrement += increment
            chunkIndices.append(totalIncrement)

        chunks = np.array_split(trace,chunkIndices)
    return chunks 

def createTemplateDictionary( trace, chunkSize):
    ''' Create a template dictionary of a given chunkSize'''
    numChunks = len(trace)/chunkSize 
    residual = len(trace)%chunkSize

    chunkArray = []
    chunkDict = {}
    for slide in range(chunkSize):

        if not (residual):
            chunks = np.split(trace,numChunks)
        else:
            chunkLength = len(trace)-residual
            chunks = np.split(trace[:chunkLength],numChunks)
            #chunks.append(trace[chunkLength:]) # The end loafs
        chunkArray.append(chunks)
        trace = np.roll(trace,-1) 
    chunkArray = np.array(map(list, zip(*chunkArray)))
    flattenedChunkArray = chunkArray.reshape((-1, chunkSize))
    chunkDict = dict(zip(range(len(flattenedChunkArray)), flattenedChunkArray))
    return chunkDict

def correlateAgainstTrace ( fragment, trace):
    ''' This uses a correlation measure to find the location of the fragment with maximum correlation to the trace.
    Has rejection criterion on both pvalue and correlation coefficients '''
    
    correlation = []
    pvalue = []
    fragment = (fragment - np.mean(fragment)) / (np.std(fragment) )
    lenFragment = len(fragment)
    for i in range( len(trace) - lenFragment +1):
        frame = trace[i:i+lenFragment]
        frame = (frame - np.mean(frame)) / (np.std(frame)* len(frame) )
        corr, pval = spearmanr(frame, fragment)
        #corr, pval = pearsonr(frame, fragment)
        #corr = np.correlate(frame,fragment )
        correlation.append(corr)
        pvalue.append(pval)
    return np.array(correlation), np.array(pvalue)

def findPeakIndexFromCorrelation(correlation, pValue = [], correlation_tolerance = 0.90, pValue_tolerance = 0.05, onlyBestMatch=False):

    ''' Finding indices of peaks from indices '''
    indices = np.where( correlation > correlation_tolerance )
    if pValue.any():
        indices = indices[0][np.where(pValue[indices] < pValue_tolerance)]
        if onlyBestMatch:
            newIndices = np.where(correlation[indices] == max(correlation[indices]))
        else:
            newIndices = indices
    else:
        newIndices = indices

    return newIndices

def findFragment(fragment, trace, t, tmax, fragmentBoundary=[], onlyBestMatch = False ):
    ''' Locates a fragment in a trace and its other occurences given one trace 
    and the boundary of the fragment, or in another trace, given the fragment itself. '''

    if np.any(fragmentBoundary):
       fragmentIndices = range(fragmentBoundary[0],fragmentBoundary[1] )
       fragment = trace[fragmentIndices]
       tfrag = t[fragmentIndices] 

    correlation, pvalue = correlateAgainstTrace ( fragment, trace)
    tcorr = np.linspace(0, float(tmax), len(correlation))
    if len(tcorr) > len(correlation):
        tcorr = tcorr[:-1]
    indices = findPeakIndexFromCorrelation(correlation, pValue = pvalue, onlyBestMatch = onlyBestMatch)

    return tcorr, fragment, correlation, indices

def pickleObject(obj, name ):
    ''' Save object as a pickle '''
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def unpickleObject(name ):
    ''' Load pickled object '''
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def makeTemplateDictionaryFromFile(filename, wordLength, acquisitionFrequency, blankIndices, fullTemplateIndices, bandPass):
    ''' De-noise a sequence, and create a template dictionary for searching the main trace '''

    photodiodeDict, null  = util.parseDictKeys(filename)
    photodiodeList = []
    for key in photodiodeDict.keys():
        if key==2:
            photodiodeDict[key] = photodiodeDict[key].T
            photodiodeList.extend(photodiodeDict[key][1])
     
    totalTime = (len(photodiodeList)/acquisitionFrequency)*1000. # In ms
    print len(photodiodeList)
    #plt.plot(np.arange(0,totalTime,1000./acquisitionFrequency), photodiodeList)
    #plt.title('template')
    #Spectrum, Filtered_spectrum, Filtered_signal_1, Low_point, High_point = util.bandpass_ifft(np.array(photodiodeList), 30, 90, acquisitionFrequency, nfactor= downsampleBy)
    Spectrum, Filtered_spectrum, Filtered_signal_1, Low_point, High_point = util.bandpass_ifft(np.array(photodiodeList), bandPass[0], bandPass[1], acquisitionFrequency )
    print len(Filtered_signal_1)
    
    ################## A single blank slice ####################################
    blank = Filtered_signal_1[blankIndices[0]:blankIndices[1]] 
    #plt.plot(np.arange(0,(len(blank)/acquisitionFrequency)*1000, 1000./acquisitionFrequency), blank)
    #plt.show()
    blank_trace, maxBlankIndices, minBlankIndices = filterByExtremas(blank)
    blank_trace = np.array(blank_trace).flatten()/np.array(blank_trace).max() # Normalizing to get normalized correlations.

    ################## A single template slice ####################################
    singleTrace = Filtered_signal_1[fullTemplateIndices[0]:fullTemplateIndices[1]]
    photodiode_singlesignal = photodiodeList[fullTemplateIndices[0]:fullTemplateIndices[1]]
    single_trace, maxIndices, minIndices = filterByExtremas(singleTrace)
    reducedSignal = np.array(single_trace).flatten()/np.array(single_trace).max() # Normalizing to get normalized correlations.
    library = createTemplateDictionary( single_trace, wordLength)

    filteredIndices = sorted(np.append(maxIndices, minIndices))
    print len(maxIndices[0]), len(minIndices[0])
    print len(filteredIndices)
    return library, blank_trace, reducedSignal, photodiodeList, Filtered_signal_1, filteredIndices 

def makeTraceDictionaryFromFile(filename, wordLength, acquisitionFrequency, bandPass, downSampleBy = 1):
    ''' De-noise a sequence, and create a trace dictionary '''

    voltage, photodiodeDict = util.parseDictKeys(filename)
    if not photodiodeDict:
        photodiodeDict, null = util.parseDictKeys(filename)
        
    photodiodeList = []
    for key in photodiodeDict.keys():
            photodiodeDict[key] = photodiodeDict[key].T
            photodiodeList.extend(photodiodeDict[key][1])

    totalTime = (len(photodiodeList)/acquisitionFrequency)*1000. # In ms
    Spectrum, Filtered_spectrum, Filtered_signal_1, Low_point, High_point = util.bandpass_ifft(np.array(photodiodeList), bandPass[0], bandPass[1], acquisitionFrequency, nfactor= downSampleBy)
 
    #Spectrum, Filtered_spectrum, Filtered_signal_1, Low_point, High_point = util.bandpass_ifft(np.array(photodiodeList), 30, 90, acquisitionFrequency )
    
    ################## A single template slice ####################################
    single_trace, maxIndices, minIndices = filterByExtremas(Filtered_signal_1)
    signal = np.array(single_trace).flatten()/np.array(single_trace).max() # Normalizing to get normalized correlations.
    library = createTemplateDictionary( single_trace, wordLength)

    return library, signal, Filtered_signal_1, sorted(np.append(maxIndices, minIndices))

def processPDSignalFromFile(filename, acquisitionFrequency, bandPass, downSampleBy = 1):
    ''' De-noise a sequence, and process to give the trace '''
    voltage, photodiodeDict = util.parseDictKeys(filename)
    if not photodiodeDict:
        photodiodeDict, null = util.parseDictKeys(filename)
        
    photodiodeList = []

    for key in photodiodeDict.keys():
        if key == 5:
            photodiodeDict[key] = photodiodeDict[key].T
            photodiodeList.extend(photodiodeDict[key][1])

    totalTime = (len(photodiodeList)/acquisitionFrequency)*1000. # In ms
    Spectrum, Filtered_spectrum, Filtered_signal_1, Low_point, High_point = util.bandpass_ifft(np.array(photodiodeList), bandPass[0], bandPass[1], acquisitionFrequency, nfactor= downSampleBy)
    #Spectrum, Filtered_spectrum, Filtered_signal_1, Low_point, High_point = util.bandpass_ifft(np.array(photodiodeList), 30, 90, acquisitionFrequency )
 
    ######################### Full trace #######################################
    filteredSignal, maxIndices, minIndices = filterByExtremas(Filtered_signal_1)
    
    trace = np.array(filteredSignal).flatten()/np.array(filteredSignal).max() # Normalizing to get normalized correlations.
    return trace,Filtered_signal_1, photodiodeList, totalTime , sorted(np.append(maxIndices, minIndices))

def calculateReliability(expectedNumer, actualNumber):
    reliability = np.exp( - (1 - ( actualNumber / float(expectedNumber) ) ) ** 2 )
    return reliability 

def thresholdPeaksByReliability ( indexingDict, thresholdReliability, expectedNumber ):
    peakReliability = {}
    for key in indexingDict.keys():
        if len(indexingDict[key]):
            peakReliability.update({ key: calculateReliability(expectedNumber, len(indexingDict[key])) })
        else:
            peakReliability.update({ key: 0. })
    
    for key in peakReliability.keys():
        if peakReliability[key] < thresholdReliability:
            indexingDict[key] = []
    return indexingDict

def findLongestIncrementingSubSequence (sequenceDict, wordLength, retrieveSorted = 0):
    ''' Finds the longest incrementing subequence from the sets given here '''
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

    for key in globalLiveSequences.keys():
        globalLiveSequences[key]+=wordLength

    if (retrieveSorted):
        retrievedList = sorted(globalLiveSequences.items(), key=lambda x:x[1], reverse=True)[:retrieveSorted] 
    else:
        retrievedList = sorted(globalLiveSequences.items(), key=lambda x:x[1], reverse=True)
 
    return retrievedList, globalLiveSequences

def assignCoords( coords, numSquares,  pdTrace, totalFrames ):
    ''' Assigns corrds to the template trace, so that the unknown trace can be mapped accordingly'''
    x,y = coords
    xy = it.izip(x,y) # This will keep doing an indefinite cycling through the coordinates.
    outputVec = []
    projectionDict = {}
    templateMap = {}
    chunkedArray = chunkTrace(pdTrace, numChunks = totalFrames)
    for frameNum in range(totalFrames):
       projectionDict[frameNum] = []
       for i in range(numSquares):
           xValue, yValue = xy.next()
           projectionDict[frameNum].append((xValue,yValue))
       templateMap.update({frameNum: chunkedArray[frameNum]})

    return projectionDict, templateMap

def assignCoordsToUnknownTrace(listOfPredictedSequences, coordinateTemplateMap, traceLength):
    ''' This function assigns coords using the template map and makes prediction for the entire trace '''
    matchedCoords = []
    for i in range(len(listOfPredictedSequences)):
        templateIndex = int(listOfPredictedSequences[i][0][0])
        traceIndex = int(listOfPredictedSequences[i][0][1])
        lengthOfSequence = int(listOfPredictedSequences[i][1])
        for i in range(lengthOfSequence):
            matchedCoords.append(coordinateTemplateMap[templateIndex+i])

        numPatterns = len ( coordinateTemplateMap.keys())
        startingIndex = numPatterns - (traceIndex%numPatterns)

        preMatch =  []
        postMatch = [] 

        i = 0 # Count
        mappingIndices = [] ## Just to map the starting of the trace with the template index.
        for k in it.cycle(sorted(coordinateTemplateMap.keys())):   ## Cycling through the template map
            if i >= startingIndex:
                if i< ( startingIndex + traceIndex ):
                    preMatch.append(coordinateTemplateMap[k])
                elif (i>= startingIndex + traceIndex + lengthOfSequence) and (i<startingIndex + traceLength):
                    postMatch.append(coordinateTemplateMap[k])
                elif i>=startingIndex + traceLength:
                    break
                mappingIndices.append(k)
            else:
                pass
            i+=1
        print "lengths are "
        print len(preMatch), len(matchedCoords), len(postMatch)
        finalCoords =  preMatch + matchedCoords + postMatch
        return finalCoords, mappingIndices

###############################################################################

### Reading random stimulation coordinates for 5 square stimulation from file here ###

randX = '/home/sahil/Documents/Codes/bgstimPlasticity/data/august/data_RS_aug/randX.txt'
randY = '/home/sahil/Documents/Codes/bgstimPlasticity/data/august/data_RS_aug/randY.txt'
coords = util.readBoxCSV(randX,randY)

######################## Reading MATLAB photodiode traces #######################################################

#photodiode_template = util.readMatlabFile('/home/sahil/Documents/Codes/bgstimPlasticity/data/august/projectorTraces/ 150903/250int_20khz.mat')
photodiode_template = util.readMatlabFile('/home/sahil/Documents/Codes/bgstimPlasticity/photoDiode/160205_projector/Sahils_code_v3/10sec.mat')
photodiode_unknown = util.readMatlabFile('/home/sahil/Documents/Codes/bgstimPlasticity/photoDiode/160205_projector/Sahils_code_v3/10sec.mat')

###################  Template  #####################################################

templateBlankIndices = [8564,12564]
fullTemplateIndices = [12564,260000]
lengthFullTemplate = fullTemplateIndices[1] - fullTemplateIndices[0]  
lengthBlank =  templateBlankIndices[1] - templateBlankIndices[0] 

#chunks, anomaly  = chunkTrace( filteredSignal, 3)
#print chunks, anomaly

totalFrames = 900
numSquares = 5
wordLength = 15

acquisitionFrequency = 20000.
timePerPoint = 1000./acquisitionFrequency # In ms

timeOfOneSequence = timePerPoint * lengthFullTemplate  # In ms
timeOfBlanks = timePerPoint * lengthBlank # In ms
timeOfOneFrame = timeOfOneSequence/totalFrames

bandPass = [0,90] ## High pass, low pass frequency.

library, blank, minMaxSignal, raw_signal, signal, templateExtremaIndices = makeTemplateDictionaryFromFile(photodiode_template, wordLength, acquisitionFrequency, templateBlankIndices, fullTemplateIndices, bandPass)

###################################### Assigning Coords to template #####################################

templateTime = np.linspace(0,timeOfOneSequence,lengthFullTemplate) # in ms
bandPassedSignal = signal[fullTemplateIndices[0]:fullTemplateIndices[1]]
templateSignal = raw_signal[fullTemplateIndices[0]:fullTemplateIndices[1]]
templateExtremaTime = templateTime[ templateExtremaIndices]

projDict, tempMap = assignCoords( coords, numSquares, templateSignal, totalFrames ) ## Coords start with zero as opposed to one!

#fig, axes  = plt.subplots()
#plt.xlim(0,np.ceil(timeOfOneFrame))
#plt.title(" Photodiode traces plotted together")
#for key in tempMap.keys():
#    axes.plot(np.linspace(0,timePerPoint*len(tempMap[key]),len(tempMap[key])), tempMap[key])
#plt.xlabel('Time (in ms) ')
#plt.ylabel('Amplitude ')
#plt.show()

#plt.plot(np.linspace(0, timeOfOneSequence, len(bandPassedSignal) ), bandPassedSignal)
#plt.plot(np.linspace(0, timeOfOneSequence, len(templateSignal) ), templateSignal)

t=0
timeDict = {}
for k in sorted(tempMap.keys()):
    timeDict.update( { k : (t, t + len(tempMap[k])*timePerPoint ) } )
#    plt.plot(np.linspace(t,t+len(tempMap[k])*timePerPoint,len(tempMap[k])), tempMap[k])
#    plt.axvline(t+len(tempMap[k])*timePerPoint)
    t += len(tempMap[k])*timePerPoint
#plt.show()

reducedTemplateCoords = {}
for i in range(len(templateExtremaTime)):
    for key in sorted(timeDict.keys()):
        if (templateExtremaTime[i] > timeDict[key][0]) and (templateExtremaTime[i] < timeDict[key][1]) :
            reducedTemplateCoords.update( { i : projDict[key] } )
#print reducedTemplateCoords.keys()

#for key,time in zip(minMaxSignal,templateExtremaTime):
    #print str(key) + ' : ' + str(time) + ' : ' + str(reducedTemplateCoords[key])

#######################################   Trace   ############################################################
acquisitionFrequency = 20000.

#minmaxtrace,trace, raw_trace, totalTime, traceExtremaIndices = processPDSignalFromFile(photodiode_unknown, acquisitionFrequency, bandPass, downSampleBy= 1)
minmaxtrace,trace, raw_trace, totalTime, traceExtremaIndices = processPDSignalFromFile(photodiode_unknown, acquisitionFrequency, bandPass)

blankIndices = locateBlank(blank, minmaxtrace)
print blankIndices

time = np.linspace(0, totalTime, len(trace))
time2 = np.linspace(0, totalTime,len(raw_trace))

minMaxTraceTime = time[traceExtremaIndices] 
np.linspace(0, totalTime, len(minmaxtrace))
print "minima is"
print len(minmaxtrace)

indexStart = 137407 
#plt.plot(time, trace, alpha=0.5, color = 'b')
plt.plot(time2, raw_trace)
plt.plot(np.arange(indexStart, indexStart + len(signal[templateBlankIndices[0]:fullTemplateIndices[1]]))*timePerPoint, raw_signal[templateBlankIndices[0]:fullTemplateIndices[1]], alpha=0.5, color = 'r')
for blankIndex in blankIndices:
    plt.axvline(x=minMaxTraceTime[blankIndex])
#plt.plot(minMaxTraceTime, minmaxtrace)
plt.show()

#######################################  Mapping sequences in Trace to template  ############################################################
indexingDict = {}

for key in library.keys():
    tcorr, fragment, correlation, indices = findFragment(library[key], minmaxtrace, minMaxTraceTime, totalTime)
    if not np.size( indices):
        indexingDict.update( {key: []} )
    else:
        indexingDict.update( {key:indices} )

pickleObject(indexingDict, 'indexedWholeTrace_RS45' )
#indexingDict = unpickleObject('indexedWholeTrace_RS45')

#expectedNumber = int(np.ceil( totalTime / (timeOfOneSequence + timeOfBlanks)))
#print expectedNumber

#expectedNumber = 4
#print expectedNumber
#thresholdReliability = 5e-2 

#indexingDict = thresholdPeaksByReliability ( indexingDict, thresholdReliability, expectedNumber )

#seqList = []
#
#for value in indexingDict.values():
#    seqList.extend(value)
#
#minVal, maxVal =  np.min(seqList), np.max(seqList)

minVal, maxVal = [0, len(minmaxtrace)] 
print "mimmaxtrace is"
print minmaxtrace

###############################################################################################################################################
sequenceList = {}

for key in range(minVal, maxVal): ## Use the actual max size instead of finding it here.
    sequenceList.update ({key: []})

for key in indexingDict.keys():
    if len(indexingDict[key]):
        for index in indexingDict[key]:
            if (index in sequenceList):
                sequenceList[index].append( key )

retrievedList, globalLiveSequences = findLongestIncrementingSubSequence (sequenceList, wordLength, retrieveSorted = 1)
print retrievedList

predictedCoordSequence,mappingIndices = assignCoordsToUnknownTrace(retrievedList, reducedTemplateCoords,len(minmaxtrace))
#print predictedCoordSequence
#print len(predictedCoordSequence), len(mappingIndices)

plt.plot(time, trace)
plt.plot(time2, raw_trace)
for blankIndex in blankIndices:
    plt.axvline(x=minMaxTraceTime[blankIndex])
plt.plot(minMaxTraceTime, minmaxtrace)

j=0
xaxis = []
yaxis = []
for i in range(len(mappingIndices)):
    if j < len(minMaxTraceTime):
       if mappingIndices[i]==totalFrames:
           plt.axvline(x=minMaxTraceTime[j], color = 'y')
           #print "yay!"
           xaxis.extend(minMaxTraceTime[j:j+len(blank)])
           yaxis.extend(blank.tolist())
           #plt.plot(minMaxTraceTime[j:j+len(blank)], blank, 'black')
           j+=len(blank)
       else:
           print minMaxTraceTime[j], minMaxSignal[mappingIndices[i]]
           xaxis.append(minMaxTraceTime[j])
           yaxis.append(minMaxSignal[mappingIndices[i]])
           #plt.plot(minMaxTraceTime[j], minMaxSignal[mappingIndices[i]],'-o')
           j+=1
print len(xaxis), len(yaxis)
plt.plot(xaxis, yaxis, 'black')
plt.show()


#correlation = {}
#for chunk in chunks:
#    for key in library.keys():
#        find 
#    normalizingFactor = 1./(np.array(correlation.values()).max())
#    for key in correlation.keys():
#        correlation[key] = np.array(correlation[key]).flatten()/normalizingFactor
#        if correlation[key] >= 0.999:
#            ''' Attach the key to the chunk '''
#            ''' Also, I need a unique dictionary for the random stimulation, so that stuff can be uniquely mapped '''
