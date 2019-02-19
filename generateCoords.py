#!/usr/bin/python2.7
import sys
import Linearity 
import os
from util import getInputSizeOfPhotoActiveGrid, readBoxCSV, readMatlabFile, parseDictKeys, find_BaseLine_and_WindowOfInterest_Margins
import matplotlib.pyplot as plt
plt.style.use('neuron')
from analysisVariables import *
import scipy.stats as ss
import numpy as np
import random
import Tkinter, tkMessageBox, tkFileDialog
import itertools

numEdgeSquares = 13
circleDiameter = 10
skipSquaresBy = 2
diagonalOnly = False
numSquareRepeats = 1

def simpleaxis(axes, every=False, outward=False):
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if (outward):
            ax.spines['bottom'].set_position(('outward', 10))
            ax.spines['left'].set_position(('outward', 10))
        else:
            ax.spines['bottom'].set_position(('outward', 10))
            ax.spines['left'].set_position(('outward', 10))

        if every:
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_title('')

def makePhotoStimulationGrid(numEdgeSquares, skipSquaresBy=1):
    photoStimGrid = []
    k = True
    for i in xrange(0,numEdgeSquares):
        k = not k # To diagonally flip between lines
        for j in xrange(int(k),numEdgeSquares,skipSquaresBy):
            photoStimGrid.append((i,j))
    return photoStimGrid 

def makeCircularPhotostimulationGrid(numEdgeSquares,circleDiameter,skipSquaresBy=1, diagonalOnly=False):
    photoStimGrid = makePhotoStimulationGrid(numEdgeSquares,skipSquaresBy=skipSquaresBy)
    circularPhotoStimGrid = []
    photoStimCopy = np.zeros((numEdgeSquares,numEdgeSquares))
    circleCenter = np.ceil(numEdgeSquares/2.)
    print circleCenter
    for index in photoStimGrid:
        if not diagonalOnly:
            if ( np.floor( ((index[0]-circleCenter)**2) + ((index[1]-circleCenter)**2)) <= 0.25*(circleDiameter**2) ):
                photoStimCopy[(index)] = 1
                circularPhotoStimGrid.append(index)
        else:
            if index[0] == index[1]:
                if ( np.floor( ((index[0]-circleCenter)**2) + ((index[1]-circleCenter)**2)) <= 0.25*(circleDiameter**2) ):
                    photoStimCopy[(index)] = 1
                    circularPhotoStimGrid.append(index)
                print "Index:{}".format(index)    

    x,y = returnCoordsFromGrid(circularPhotoStimGrid) 

    jointX = []
    jointY = []
    for k in range(len(x)):
        jointX.append(x[k*len(circularPhotoStimGrid):(k+1)*len(circularPhotoStimGrid)])
    
    for k in range(len(y)):
        jointY.append(y[k*len(circularPhotoStimGrid):(k+1)*len(circularPhotoStimGrid)])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x,y,c='k')
    ax.set_xlim(0,numEdgeSquares)
    ax.set_ylim(0,numEdgeSquares)
    simpleaxis(ax)
    fig.set_figwidth(2)
    fig.set_figheight(2)
    plt.show()
    return circularPhotoStimGrid 
            
def createRandomPhotoStimulation(totalNumberOfSquares, photoStimGrid):
    totalIterations = int(np.ceil( float(totalNumberOfSquares)/ len(photoStimGrid)))
    repeatedPhotoStimGrid = []
    for j in range(totalIterations):
        np.random.shuffle(photoStimGrid)
        repeatedPhotoStimGrid.extend(photoStimGrid)
    residualIndices = totalNumberOfSquares % len(photoStimGrid)
    if residualIndices:
        repeatedPhotoStimGrid = repeatedPhotoStimGrid[:((totalIterations-1)*len(photoStimGrid)) + residualIndices]
    return repeatedPhotoStimGrid

def returnCoordsFromGrid(photoStimGrid):
    xcoords = []
    ycoords = []
    for coords in photoStimGrid:
        if type(coords[0])==int:
            xcoords.append(coords[0])
            ycoords.append(coords[1])
        else:
            for m in coords: 
                xcoords.append(m[0])
                ycoords.append(m[1])
    return xcoords, ycoords

def createPhotoStimulation_init(outDirectory): 
    print len(circularGrid), totalNumberOfSquares
    circularRandomStimulationGrid = createRandomPhotoStimulation(totalNumberOfSquares, circularGrid)
    print len(circularRandomStimulationGrid)
    
    x,y = returnCoordsFromGrid(circularRandomStimulationGrid) 
    
    with open(os.path.join(outDirectory , "randX.txt"),'w') as coordFile:
        coordFile.write(','.join( [str(i) for i in x] ))
    
    with open(os.path.join(outDirectory , "randY.txt"),'w') as coordFile:
        coordFile.write(','.join( [str(i) for i in y] ))
    
    with open(os.path.join(outDirectory , "CPP_randX.txt"),'w') as coordFile:
        coordFile.write(','.join( [str(i) for i in x[:len(circularGrid)]] ))
    
    with open(os.path.join(outDirectory , "CPP_randY.txt"),'w') as coordFile:
        coordFile.write(','.join( [str(i) for i in y[:len(circularGrid)]] ))
    
    jointX = []
    jointY = []
    for k in range(len(x)):
        jointX.append(x[k*len(circularGrid):(k+1)*len(circularGrid)])
    
    for k in range(len(y)):
        jointY.append(y[k*len(circularGrid):(k+1)*len(circularGrid)])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x,y,c=range(len(x)))
    ax.set_xlim(0,numEdgeSquares)
    ax.set_ylim(0,numEdgeSquares)
    simpleaxis(ax)
    fig.set_figwidth(2)
    fig.set_figheight(2)
    plt.show()

circularGrid = makeCircularPhotostimulationGrid(numEdgeSquares,circleDiameter,skipSquaresBy=skipSquaresBy, diagonalOnly=diagonalOnly)
totalNumberOfSquares = numSquareRepeats * len(circularGrid)
