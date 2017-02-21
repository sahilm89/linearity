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
        if not (os.path.exists(experimentDir + 'coords/CPP' + str(squares) + '_randX.txt') or os.path.exists(experimentDir + 'coords/CPP' + str(squares) + '_randY.txt')):
            createCoords(randX, randY, repeatSize, squares, experimentDir)

    color = iter(plt.cm.viridis(np.linspace(0, 1, len(gridSizeList))))

    for squares in gridSizeList:

        randX = experimentDir + 'coords/CPP' + str(squares) + '_randX.txt'
        randY = experimentDir + 'coords/CPP' + str(squares) + '_randY.txt'

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
        neuron.analyzeExperiment(type, numSquares, voltageTrace, photoDiode, coords, marginOfBaseLine, marginOfInterest,
                                 F_sample, smootheningTime)

        # trials = neuron.experiment[type][numSquares].trial
        # for trial in trials.values():
        #     plt.plot(trial.interestWindow)
        #     plt.vlines(trial.F_sample * trial.feature[6], min(trial.interestWindow),max(trial.interestWindow))
        #     plt.show()

if not os.path.exists(inputDir + '/plots/'):
    os.makedirs(inputDir + '/plots/')
neuron.save(inputDir + '/plots/' + index + '.pkl')
