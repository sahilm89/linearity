import sys
import Linearity
from Linearity import Experiment
import os
from util import getInputSizeOfPhotoActiveGrid, readBoxCSV, readMatlabFile, parseDictKeys,\
                 find_BaseLine_and_WindowOfInterest_Margins, createCoords
import matplotlib.pyplot as plt
from analysisVariables_CPA import *
import numpy as np
import copy
import pickle

def parseDictKeys(fullDict):
    ''' Converts the dictionary from the matlab file into different dicts of 
    photodiode traces and voltage traces from the CA1'''
    photoDiode = {}
    photoDiodeFull = {}
    voltageTrace = {}
    currentType = {}

    index = 1
    flag1 = 1
    i = 1
    totIndex = 0

    while(flag1):
        j = 1
        flag1 = 0
        flag2 = 1
        while(flag2):
           key = 'Trace_1_' + str(i) + '_' + str(j) + '_1'
           if (key in fullDict):
               flag1 = 1
               voltageTrace.update({totIndex + j: fullDict[key]})
               if 'Trace_1_' + str(i) + '_' + str(j) + '_2' in fullDict:
                   photoDiode.update({totIndex + j: fullDict['Trace_1_' + str(i) + '_' + str(j) + '_2']})
               j+=1
           else:
               flag2 = 0
               totIndex = 0  
        currentType[i] = copy.deepcopy(voltageTrace)
        photoDiodeFull[i] = copy.deepcopy(photoDiode)
        i+=1
    return currentType, photoDiodeFull 

def analyzeExperiment(instance, type, squares, voltage, photodiode, coords,
                      marginOfBaseLine, marginOfInterest,
                      F_sample, smootheningTime, filtering=''):
    instance.filtering = filtering
    instance.removeAP = False
    if type not in instance.experiment:
        instance.experiment[type] = {squares: Experiment(instance, type, squares,
                                 voltage, photodiode, coords,
                                 marginOfBaseLine, marginOfInterest,
                                 F_sample, smootheningTime,repeatCoords=True)}
    else:
        instance.experiment[type].update({squares: Experiment(instance, type,
                                      squares, voltage, photodiode,
                                      coords, marginOfBaseLine,
                                      marginOfInterest, F_sample,
                                      smootheningTime,repeatCoords=True)})

    instance.experiment[type][squares]._groupTrialsByCoords()  # Coord grouping

def setup(inputDir, index, date, save_trial=False):
    neuron = Linearity.Neuron(index, date, save_trial)
    print neuron.index, neuron.date
    
    experimentDir = inputDir + '/' + 'CPA/'
    
    gridSizeList = [''] + getInputSizeOfPhotoActiveGrid(experimentDir)
    
    if os.path.exists(experimentDir + 'coords/filtered_CPA_randX.txt'):
        randX = experimentDir + 'coords/filtered_CPA_randX.txt'
        randY = experimentDir + 'coords/filtered_CPA_randY.txt'
    else:
        randX = experimentDir + 'coords/CPA_randX.txt'
        randY = experimentDir + 'coords/CPA_randY.txt'
    
    coords = readBoxCSV(randX, randY)
    repeatSize = len(coords[0])

    randX = experimentDir + 'coords/randX.txt'
    randY = experimentDir + 'coords/randY.txt'

    for squares in gridSizeList:
        if not (os.path.exists(experimentDir + 'coords/CPA' + str(squares) + '_randX.txt') or os.path.exists(experimentDir + 'coords/CPA' + str(squares) + '_randY.txt')):
                createCoords(randX, randY, repeatSize, squares, experimentDir)
        sqr_randX = experimentDir + 'coords/CPA' + str(squares) + '_randX.txt'
        sqr_randY = experimentDir + 'coords/CPA' + str(squares) + '_randY.txt'
        coords = readBoxCSV(sqr_randX, sqr_randY)
        assert len(coords[0]) == len(coords[1]), "{},{}".format(len(coords[0]), len(coords[1]))
        coords_1 = [tuple(crd) for n, crd in enumerate(zip(coords[0], coords[1])) if n%2 == 0]
        coords_2 = [tuple(crd) for n, crd in enumerate(zip(coords[0], coords[1])) if n%2 == 1]
        
        print (coords_1)
        print (coords_2)
    
        CPA = experimentDir + 'CPA' + str(squares) + '.mat'
        if os.path.exists(CPA):
            fullDict = readMatlabFile(CPA)
            currentType, photoDiode = parseDictKeys(fullDict)
    
            if not squares:
                numSquares = 1  # Prefix for CPA1
            else:
                numSquares = squares
    
            print numSquares, currentType.keys()
            for type in currentType:
                marginOfBaseLine, marginOfInterest_1 = find_BaseLine_and_WindowOfInterest_Margins(photoDiode[type], threshold,
                                baselineWindowWidth, interestWindowWidth)
                marginOfInterest_2 = (marginOfInterest_1[1], marginOfInterest_1[1] + interestWindowWidth)
                analyzeExperiment(neuron, (type,1), numSquares, currentType[type], photoDiode[type], coords_1, marginOfBaseLine, marginOfInterest_1,
                        F_sample, smootheningTime, filtering = 'bessel')
                analyzeExperiment(neuron, (type,2), numSquares, currentType[type], photoDiode[type], coords_2, marginOfBaseLine, marginOfInterest_2,
                        F_sample, smootheningTime, filtering = 'bessel')
        else:
            print "File not found!" 

    return neuron

inputDir = os.path.abspath(sys.argv[1])
index, date = inputDir.split('/')[::-1][:2]

plotFile = inputDir + '/plots/' + index + '.pkl'
neuron = setup(inputDir, index, date, save_trial=True)
neuron.save(plotFile)

with open(plotFile, 'rb') as input:
    neuron = pickle.load(input)

inhib = []
excit = []
onset_e = []
onset_i = []

for type in neuron.experiment.keys():
    for square in neuron.experiment[type]:
        currentVals = neuron.experiment[type][square]
        print type
        if type == 1:
            for coord in currentVals.coordwise:
                excit.append(abs(currentVals.coordwise[coord].average_feature[0]))
                onset_e.append(currentVals.coordwise[coord].average_feature[6])
        elif type == 2:
            for coord in currentVals.coordwise:
                inhib.append(abs(currentVals.coordwise[coord].average_feature[0]))
                onset_i.append(currentVals.coordwise[coord].average_feature[6])

excit, inhib =  1e6*np.array(excit), 1e6*np.array(inhib) # Already in mA
onset_e, onset_i = np.array(onset_e), np.array(onset_i)
print onset_i, onset_e

delay,excitation = [], []
for eon, ion, e in zip(onset_e, onset_i, excit):
    #if not (eon == 0. or ion == 0.): 
        delay.append(ion - eon)
        excitation.append(e)
print delay

fig,ax = plt.subplots()
ax.scatter(excit, inhib)
ylim=ax.get_ylim()
xlim= (0, ylim[1])
ylim= (0, ylim[1])
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig(inputDir + '/plots/' + 'EI.png')

#print excitation, delay
#plt.scatter(excitation, delay)
#plt.xlabel("Excitation")
#plt.ylabel("Delay")
#plt.show()
#
if not os.path.exists(inputDir + '/plots/'):
    os.makedirs(inputDir + '/plots/')
neuron.save(inputDir + '/plots/' + index + '_CPA.pkl')
