#!/usr/bin/python2.7
import sys
import Linearity 
import os
from util import getInputSizeOfPhotoActiveGrid, readBoxCSV, readMatlabFile, parseDictKeys, find_BaseLine_and_WindowOfInterest_Margins, createCoords
import matplotlib.pyplot as plt
from analysisVariables import *
import scipy.stats as ss
import numpy as np
import Tkinter, tkMessageBox, tkFileDialog

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
        xcoords.append(coords[0])
        ycoords.append(coords[1])
    return xcoords, ycoords

def createPhotoStimulation_init(outDirectory): 
    print len(circularGrid), totalNumberOfSquares
    circularRandomStimulationGrid = createRandomPhotoStimulation(totalNumberOfSquares, circularGrid)
    print len(circularRandomStimulationGrid)
    
    x,y = returnCoordsFromGrid(circularRandomStimulationGrid) 
    
    with open(outDirectory + "/randX.txt",'w') as coordFile:
        coordFile.write(','.join( [str(i) for i in x] ))
    
    with open(outDirectory + "/randY.txt",'w') as coordFile:
        coordFile.write(','.join( [str(i) for i in y] ))
    
    with open(outDirectory + "/CPP_randX.txt",'w') as coordFile:
        coordFile.write(','.join( [str(i) for i in x[:len(circularGrid)]] ))
    
    with open(outDirectory + "/CPP_randY.txt",'w') as coordFile:
        coordFile.write(','.join( [str(i) for i in y[:len(circularGrid)]] ))
    
    #plt.hist(x, bins=max(x),alpha=0.5)
    
    jointX = []
    jointY = []
    for k in range(len(x)):
        jointX.append(x[k*len(circularGrid):(k+1)*len(circularGrid)])
    
    for k in range(len(y)):
        jointY.append(y[k*len(circularGrid):(k+1)*len(circularGrid)])
    
    #plt.hist(jointX,bins=max(x),alpha=0.5,stacked=True)
    #plt.hist(x,bins=max(x),alpha=0.5)
    #plt.show()
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(x,y,c=range(len(x)))
    #ax.set_xlim(0,numEdgeSquares)
    #ax.set_ylim(0,numEdgeSquares)
    #plt.show()


def createCoordinatesFromOneSquareData(inputDir, plotResponse=False):
    inputDir = os.path.abspath(inputDir)
    index, date = inputDir.split('/')[::-1][:2]
    
    neuron = Linearity.Neuron(index, date)
    print neuron.index, neuron.date 
    type = 'Control'
    experimentDir = inputDir + '/CPP/'
    
    randX = experimentDir + 'coords/CPP_randX.txt'
    randY = experimentDir + 'coords/CPP_randY.txt'
    
    coords = readBoxCSV(randX,randY)
    repeatSize = len(coords[0])
    
    randX = experimentDir + 'coords/randX.txt'
    randY = experimentDir + 'coords/randY.txt'
    
    CPP = experimentDir + 'CPP.mat'
    #for squares in gridSizeList:
    #    createCoords(randX, randY, repeatSize, squares, experimentDir)
    
    numSquares = 1
    assert len(coords[0]) == len(coords[1]), "{},{}".format(len(coords[0]), len(coords[1]))
    coords = [(i,j) for i,j in zip(coords[0], coords[1])]
    
    fullDict = readMatlabFile(CPP)
    voltageTrace, photoDiode = parseDictKeys(fullDict)
    marginOfBaseLine, marginOfInterest = find_BaseLine_and_WindowOfInterest_Margins(photoDiode,threshold, baselineWindowWidth, interestWindowWidth)
    neuron.analyzeExperiment(type, numSquares, voltageTrace, photoDiode, coords, marginOfBaseLine, marginOfInterest,F_sample, smootheningTime )
    
    coordwise = neuron.experiment[type][1].coordwise
    vmax_dict = {}
    for coord in coordwise:
        vmax_dict.update({list(coordwise[coord].coords)[0]: coordwise[coord].average_feature[0]})
    
    print sorted(vmax_dict, key=vmax_dict.get)
    threshold_voltage = 5e-4
    coord_list = [key for key in sorted(vmax_dict, key=vmax_dict.get) if vmax_dict[key]>threshold_voltage]
    print len(coord_list)
    coord_list = coord_list[-24:]
    print len(coord_list)

    circularRandomStimulationGrid = createRandomPhotoStimulation(numSquareRepeats*len(coord_list), coord_list)
    x,y = returnCoordsFromGrid(circularRandomStimulationGrid) 

    prefix = "/media/sahil/InternalHD/170208/c1/CPP/"

    with open(experimentDir + "coords/randX.txt",'w') as coordFile:
        coordFile.write(','.join( [str(i+1) for i in x] ))
    
    with open(experimentDir + "coords/randY.txt",'w') as coordFile:
        coordFile.write(','.join( [str(i+1) for i in y] ))
    
    with open(experimentDir + "coords/filtered_CPP_randX.txt",'w') as coordFile:
        coordFile.write(','.join( [str(i+1) for i in x[:len(coord_list)]] ))
    
    with open(experimentDir + "coords/filtered_CPP_randY.txt",'w') as coordFile:
        coordFile.write(','.join( [str(i+1) for i in y[:len(coord_list)]]))
    
    #plt.hist(x, bins=max(x),alpha=0.5)
    
    jointX = []
    jointY = []
    for k in range(len(x)):
        jointX.append(x[k*len(circularGrid):(k+1)*len(coord_list)])
    
    for k in range(len(y)):
        jointY.append(y[k*len(circularGrid):(k+1)*len(coord_list)])
    
    #plt.hist(jointX,bins=max(x),alpha=0.5,stacked=True)
    #plt.hist(x,bins=max(x),alpha=0.5)
    #plt.show()

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(x,y,c=range(len(x)))
    #ax.set_xlim(0,numEdgeSquares)
    #ax.set_ylim(0,numEdgeSquares)
    #plt.show()

    if plotResponse:

        vmax_dict_new = { key:vmax_dict[key] for key in coord_list}
        plt.hist(vmax_dict_new.values())
        plt.xlabel("$V_{max}$")
        plt.show()

class App():
    def __init__(self):
        self.root = Tkinter.Tk()
        self.root.geometry("500x200")
        self.root.wm_title("Coordinate Generator")
        self.plotResponse = 0

        labelframe = Tkinter.LabelFrame(self.root, text="Generate new coordinates ")
        labelframe.pack(fill="both", expand="yes")
        left = Tkinter.Button(labelframe, text = 'Generate randX from scratch', command=self.generateRandXFromScratch)
        left.pack()

        labelframe_2 = Tkinter.LabelFrame(self.root, text="Create randX from CPP values ")
        labelframe_2.pack(fill="both", expand="yes")
        C1 = Tkinter.Checkbutton(labelframe_2, text = "Plot Response", variable = self.plotResponse, onvalue = 1, offvalue = 0, height=5, width = 20)
        right = Tkinter.Button(labelframe_2, text = 'Give location of cell', command=self.generateRandXFromData)
        C1.pack()
        right.pack()

        button = Tkinter.Button(self.root, text = 'Close', command=self.quit)
        button.pack()
        self.root.mainloop()

    def quit(self):
        self.root.destroy() 

    def generateRandXFromScratch(self):
        dir = self.get_dir()
        if not dir:
            return
        else:
            createPhotoStimulation_init(dir)

    def get_dir(self):
        dir = tkFileDialog.askdirectory(initialdir='.')
        return dir

    def generateRandXFromData(self):
        dir = self.get_dir()
        if not dir:
            return
        else:
            createCoordinatesFromOneSquareData(dir, plotResponse=self.plotResponse)

numEdgeSquares = 13
circleDiameter = 10
skipSquaresBy = 2
diagonalOnly = False
circularGrid = makeCircularPhotostimulationGrid(numEdgeSquares,circleDiameter,skipSquaresBy=skipSquaresBy, diagonalOnly=diagonalOnly)
numSquareRepeats = 10
totalNumberOfSquares = numSquareRepeats * len(circularGrid)

app = App()
