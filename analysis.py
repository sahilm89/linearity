import sys
import Linearity
import os
from util import getInputSizeOfPhotoActiveGrid, readBoxCSV, readMatlabFile, parseDictKeys,\
                 find_BaseLine_and_WindowOfInterest_Margins, createCoords
import matplotlib.pyplot as plt
from analysisVariables import *
import numpy as np

inputDir = os.path.abspath(sys.argv[1])
index, date = inputDir.split('/')[::-1][:2]
numCoords = 24

neuron = Linearity.Neuron(index, date)
print neuron.index, neuron.date

for type in ['Control', 'GABAzine']:

    if type == 'Control':
        experimentDir = inputDir + '/' + 'CPP/'
    else:
        experimentDir = inputDir + '/' + type + '/' + 'CPP/'
    if not os.path.exists(experimentDir):
        break

    gridSizeList = [''] + getInputSizeOfPhotoActiveGrid(experimentDir)

    if os.path.exists(experimentDir + 'coords/filtered_CPP_randX.txt'):
        randX = experimentDir + 'coords/filtered_CPP_randX.txt'
        randY = experimentDir + 'coords/filtered_CPP_randY.txt'
    else:
        randX = experimentDir + 'coords/CPP_randX.txt'
        randY = experimentDir + 'coords/CPP_randY.txt'

    coords = readBoxCSV(randX, randY)
    repeatSize = len(coords[0])

    randX = experimentDir + 'coords/randX.txt'
    randY = experimentDir + 'coords/randY.txt'

    for squares in gridSizeList[1:]:
        print squares
        if not (os.path.exists(experimentDir + 'coords/CPP' + str(squares) + '_randX.txt') or os.path.exists(experimentDir + 'coords/CPP' + str(squares) + '_randY.txt')):
            print "Creating coordinates for {} squares".format(squares)
            createCoords(randX, randY, repeatSize, squares, experimentDir)

    color = iter(plt.cm.viridis(np.linspace(0, 1, len(gridSizeList))))

    for squares in gridSizeList:

        randX = experimentDir + 'coords/CPP' + str(squares) + '_randX.txt'
        randY = experimentDir + 'coords/CPP' + str(squares) + '_randY.txt'

        #coords = readBoxCSV(randX, randY, length=squares*numCoords) # Why was this here?
        coords = readBoxCSV(randX, randY)
        # print [coord for coord in zip(coords[0], coords[1])]
        assert len(coords[0]) == len(coords[1]), "{},{}".format(len(coords[0]), len(coords[1]))
        coords = [(i, j) for i, j in zip(coords[0], coords[1])]

        CPP = experimentDir + 'CPP' + str(squares) + '.mat'

        if not squares:
            numSquares = 1  # Prefix for CPP1
        else:
            numSquares = squares

        print "\n Reading from {} ({} squares)".format(CPP, numSquares)

        fullDict = readMatlabFile(CPP)
        voltageTrace, photoDiode = parseDictKeys(fullDict)
        marginOfBaseLine, marginOfInterest = find_BaseLine_and_WindowOfInterest_Margins(photoDiode, threshold,
                baselineWindowWidth, interestWindowWidth)
        print (inputDir.split('/')[-1])
        if any(x in inputDir.split('/')[-1] for x in ['CS', 'spikes, CA3_CPP']):
            neuron.analyzeExperiment(type, numSquares, voltageTrace, photoDiode, coords, marginOfBaseLine, marginOfInterest,
                                 F_sample, smootheningTime, removeAP=False)
        else:
            neuron.analyzeExperiment(type, numSquares, voltageTrace, photoDiode, coords, marginOfBaseLine, marginOfInterest,
                                 F_sample, smootheningTime, removeAP=True, filtering=filtering)

if not os.path.exists(inputDir + '/plots/'):
    os.makedirs(inputDir + '/plots/')

fig, ax = plt.subplots()
for expType, exp in neuron:
    if expType == "Control":
        for sqr in exp:
            if sqr>1:
                list_exp, list_obs = [], []
                for trial in exp[sqr].trial:
                    if (0 in exp[sqr].trial[trial].expected_feature) and (0 in exp[sqr].trial[trial].feature):
                        list_exp.append(exp[sqr].trial[trial].expected_feature[0])
                        list_obs.append(exp[sqr].trial[trial].feature[0])
                ax.scatter(list_exp, list_obs)
xlim = ax.get_xlim()
xlim = (0,xlim[1])
ax.set_xlim(xlim)
ax.set_ylim(xlim)
plt.savefig(inputDir + '/plots/' + 'E_O.png')

neuron.save(inputDir + '/plots/' + index + '.pkl')
print "Wrote {}".format(inputDir + '/plots/' + index + '.pkl')
