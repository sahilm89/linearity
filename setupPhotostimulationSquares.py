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

np.random.seed = randSeed 
random.seed = randSeed 

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

def sampleCoordinates(oneSquareDictionary, number, square = 1, mode='uniform', threshold_voltage=0.5, max_voltage=30, maxCombinations=1e6): # Units are mV
    ''' Samples coordinates from the dictionary of coordinates and values provided, uniformly or largest'''

    if mode == 'uniform':
        #coordinateDict =  {}
        coord_list_pre = []
        coord_values_pre = []
        comb = itertools.combinations(oneSquareDictionary, square)
        numCombs = 0
        for j in comb:
            sumValue = sum([oneSquareDictionary[k] for k in j])
            if ( sumValue > max_voltage) or (sumValue < threshold_voltage): 
                continue
            else:
                coord_list_pre.append(j)
                coord_values_pre.append(sumValue)
                numCombs+=1
            if numCombs > maxCombinations:
                break

        coord_values_pre = np.array(coord_values_pre)
        print coord_values_pre
	
        fig, ax = plt.subplots()
        ax.hist(coord_values_pre)
        ax.set_xlabel("$V_{max}$")
        ax.set_ylabel("Frequency")
        simpleaxis(ax)
        ax.set_title(str(square) + " squares")
        fig.set_figwidth(2)
        fig.set_figheight(2)
        #plt.show()
        plt.savefig('/home/bhalla/Documents/Codes/linearity/Paper_Figures/figures/supplementary/' + str(square) + "_squares_all" + '.svg', transparent=True, bbox_inches='tight')
        plt.close()

        bins = np.linspace(np.min(coord_values_pre), np.max(coord_values_pre), number)
        bin_ids = np.digitize(coord_values_pre, bins)
        numBins = number

        while len(set(bin_ids))<number:
            bins = np.linspace(np.min(coord_values_pre), np.max(coord_values_pre), numBins)
            bin_ids = np.digitize(coord_values_pre, bins)
            numBins+=1

        sampled_bin_ids = []
        for i in set(bin_ids):
            if len(np.where(bin_ids == i)):
                sampled_bin_ids.append(random.choice(np.where(bin_ids == i)[0]))
        #while len(set(sampled_bin_ids))!=number:
        #    print "Sampling non-uniformly for coord {} to preserve number of coords to be {}".format(len(sampled_bin_ids)+1,number)
        #    random_bin = random.choice(list(set(bin_ids)))
        #    sampled_bin_ids.append(random.choice(np.where(bin_ids == random_bin)[0]))
        print "Total number of sampled coordinates for {} squares is {}".format(square, len(sampled_bin_ids))
        fig, ax = plt.subplots()
        ax.hist([coord_values_pre[j] for j in sampled_bin_ids], bins=24, color='gray')
        ax.set_xlabel("$V_{max}$")
        ax.set_ylabel("Frequency")
        simpleaxis(ax)
        ax.set_title(str(square) + " squares")
        fig.set_figwidth(2)
        fig.set_figheight(2)
        #plt.show()
        plt.savefig('/home/bhalla/Documents/Codes/linearity/Paper_Figures/figures/supplementary/' + str(square) + "_squares_sampled" + '.svg', transparent=True, bbox_inches='tight')
        plt.close()
        return [coord_list_pre[j] for j in sampled_bin_ids], np.max(coord_values_pre)

    elif mode == 'uniform_1sqr':
        coordinateDict = oneSquareDictionary
        coord_list_pre, coord_values_pre = zip(*[ (key,coordinateDict[key]) for key in sorted(coordinateDict, key=coordinateDict.get) if (coordinateDict[key]>threshold_voltage and coordinateDict[key]<max_voltage)])

        coord_values_pre = np.array(coord_values_pre)

        #plt.hist(coord_values_pre)
        #plt.title(str(square) + " squares")
        #plt.xlabel("$V_{max}$")
        #plt.ylabel("Frequency")
        #plt.show()
        #plt.close()

        bins = np.linspace(np.min(coord_values_pre), np.max(coord_values_pre), number)
        bin_ids = np.digitize(coord_values_pre, bins)
        sampled_bin_ids = []
        for i in set(bin_ids):
            if len(np.where(bin_ids == i)):
                sampled_bin_ids.append(random.choice(np.where(bin_ids == i)[0]))
        while len(sampled_bin_ids)!=number:
            print "Sampling non-uniformly for coord {} to preserve number of coords to be {}".format(len(sampled_bin_ids)+1,number)
            random_bin = random.choice(list(set(bin_ids)))
            sampled_bin_ids.append(random.choice(np.where(bin_ids == random_bin)[0]))
        print "Total number of sampled coordinates for {} squares is {}".format(square, len(sampled_bin_ids))
        #plt.hist([coord_values_pre[j] for j in sampled_bin_ids], bins=24)
        #plt.title(str(square) + " squares")
        #plt.xlabel("$V_{max}$")
        #plt.ylabel("Frequency")
        #plt.show()
        #plt.close()
        return [coord_list_pre[j] for j in sampled_bin_ids]

    elif mode == 'maximum':
        print "Initial length of coordinates ", len(coord_list_pre)
        return coord_list_pre[-number:]

def createCoordinatesFromOneSquareData(inputDir, plotResponse=False):
    inputDir = os.path.abspath(inputDir)
    index, date = inputDir.split(os.sep)[::-1][:2]
    
    neuron = Linearity.Neuron(index, date)
    print neuron.index, neuron.date 
    type = 'Control'
    experimentDir = os.path.join(inputDir , 'CPP')
    
    randX = os.path.join(experimentDir , 'coords', 'CPP_randX.txt')
    randY = os.path.join(experimentDir , 'coords', 'CPP_randY.txt')
    
    coords = readBoxCSV(randX,randY)
    repeatSize = len(coords[0])
    
    randX = os.path.join(experimentDir , 'coords', 'randX.txt')
    randY = os.path.join(experimentDir , 'coords', 'randY.txt')
    
    CPP = os.path.join(experimentDir , 'CPP.mat')
    
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
    
    threshold_voltage = 5e-4
    numCoords = 24
    squares = [2,3,5,7,9]
    print len(vmax_dict)

    fig, ax = plt.subplots()
    ax.hist(vmax_dict.values())
    ax.set_xlabel("$V_{max}$")
    ax.set_ylabel("# squares")
    simpleaxis(ax)
    fig.set_figwidth(2)
    fig.set_figheight(2)
    ax.set_title('1 square')
    plt.savefig('/home/bhalla/Documents/Codes/linearity/Paper_Figures/figures/supplementary/' + "1_squares_all" + '.svg', transparent=True, bbox_inches='tight')
    #plt.show()
    plt.close()

    for square in squares:
        coord_list, threshold_voltage = sampleCoordinates(vmax_dict, numCoords, square, threshold_voltage = threshold_voltage)
        circularRandomStimulationGrid = createRandomPhotoStimulation(numSquareRepeats*len(coord_list), coord_list)
        x,y = returnCoordsFromGrid(circularRandomStimulationGrid) 
        print len(x), len(y)

        with open(os.path.join(experimentDir , "coords", "CPP" + str(square) + "_randX.txt"),'w') as coordFile:
            coordFile.write(','.join( [str(i+1) for i in x] ))
        
        with open(os.path.join(experimentDir , "coords", "CPP" + str(square) + "_randY.txt"),'w') as coordFile:
            coordFile.write(','.join( [str(i+1) for i in y]))
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(x,y,c=range(len(x)))
    #ax.set_xlim(0,numEdgeSquares)
    #ax.set_ylim(0,numEdgeSquares)
    #plt.show()

    if plotResponse:
        vmax_dict_new = { key:vmax_dict[key] for key in coord_list}
        fig, ax = plt.subplots()
        ax.hist(vmax_dict_new.values())
        ax.xlabel("$V_{max}$")
        fig.set_figwidth(2)
        fig.set_figheight(2)
        simpleaxis(ax)
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
numSquareRepeats = 1

circularGrid = makeCircularPhotostimulationGrid(numEdgeSquares,circleDiameter,skipSquaresBy=skipSquaresBy, diagonalOnly=diagonalOnly)

totalNumberOfSquares = numSquareRepeats * len(circularGrid)

app = App()
