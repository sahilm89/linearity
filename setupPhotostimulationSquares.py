#!/usr/bin/python2.7
import sys
import Linearity 
import os
from util import getInputSizeOfPhotoActiveGrid, readBoxCSV, readMatlabFile, parseDictKeys, find_BaseLine_and_WindowOfInterest_Margins
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('neuron')
from analysisVariables import *
import scipy.stats as ss
import numpy as np
import random
import Tkinter, tkMessageBox, tkFileDialog
import itertools
from matplotlib.colors import LinearSegmentedColormap

np.random.seed = randSeed 
random.seed = randSeed 

#global maxCoords 
#maxCoords= 5

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

def generateCombinations(coordList, outDirectory):
    b = [p for p in itertools.product(coordList, repeat=2)]
    np.random.shuffle(b)
    print (b)
    
    xarr, yarr = [], []
    
    for j in b:
         x,y = zip(*j)
         xarr+=x
         yarr+=y
    
    with open(os.path.join(outDirectory , "coords", "CPA_randX.txt"),'w') as coordFile:
        coordFile.write(','.join( [str(i+1) for i in xarr] )) # +1 for the coordinate shifti from 1 instead of 0
    
    with open(os.path.join(outDirectory , "coords", "CPA_randY.txt"),'w') as coordFile:
        coordFile.write(','.join( [str(i+1) for i in yarr] )) # +1 for the coordinate shifti from 1 instead of 0

def plotGrid(neuron, maxCoords, pickedCoords=[], experimentDir='./'):
    AP_dict = np.zeros((13,13))
    SubThP_dict = np.zeros((13,13))
    for expType, exp in neuron:
        if expType == "Control":
            for coord in exp[1].coordwise:
                temp_coord = []
                temp_value = []
                for trial in exp[1].coordwise[coord].trials:
                    temp_coord.append(trial.AP_flag)
                    if not trial.AP_flag:
                        temp_value.append(trial.feature[0])
                SubThP_dict[list(coord)[0]] = np.nanmean(temp_value)        
                AP_dict[list(coord)[0]] = np.nansum(temp_coord)
    SubThP_dict = np.ma.masked_where(SubThP_dict == 0., SubThP_dict)
    print(SubThP_dict)

    vmax = np.nanmax(SubThP_dict)
    vmin = np.nanmin(SubThP_dict)
    cmap = LinearSegmentedColormap.from_list('CA3_reds', [(0., 'white'), (1., (170/256., 0, 0))])
    cmap.set_bad(color='white')

    fig, ax = plt.subplots()
    heatmap = ax.pcolormesh(SubThP_dict, cmap=cmap, vmin=vmin, vmax=vmax,picker=True)

    #kjdef onclick(event):
    #kj    x = int(np.round(event.xdata))
    #kj    y = int(np.round(event.ydata))
    #kj    tx = "Y: {}, X: {}, Value: {:.2f}".format(myylabels[y], myxlabels[x], X[y,x])
    #kj    print(tx)

    def onclick(event):
        #if not len(event.ind):
        #    return True
        #ind = event.ind[0]
        #
        #ypos=event.ind[0] / 13
        #xpos=event.ind[0] % 13
        global pickedCoords
        xpos, ypos = int(np.floor(event.xdata)), int(np.floor(event.ydata))
        print(xpos, ypos)
        circle1 = plt.Circle((xpos+0.5, ypos+0.5), 0.75, color='r', fill=False)
        ax.add_artist(circle1)
        fig.canvas.draw()
        pickedCoords.append((xpos, ypos))

        if len(pickedCoords) == maxCoords:
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
            print(pickedCoords)
            generateCombinations(pickedCoords, experimentDir)
        return

        #print("Y: "+str(self.myylabels[self.ypos])+'  X:'+str(self.myxlabels[self.xpos]))

    # mark a specific square?
    zeros = np.zeros((13,13))
    stim_coords = np.where(SubThP_dict>0)
    zeros[stim_coords] = 1.
    stim_coords = np.where(AP_dict>0)
    zeros[stim_coords] = 1.

    c = np.ma.masked_array(zeros, zeros == 0.)  # mask squares where value == 1
    ax.pcolormesh(np.arange(14), np.arange(14), c, alpha=0.5, zorder=2, facecolor='none', edgecolors='k',
                   cmap='gray', linewidth=1.)

    for y in range(AP_dict.shape[0]):
        for x in range(AP_dict.shape[1]):
            if AP_dict[y, x] > 0:
                plt.text(x + 0.5, y + 0.5, "{}".format(int(AP_dict[y, x])),
                         horizontalalignment='center',
                         verticalalignment='center', size=10)

    ax.invert_yaxis()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_ticks(np.arange(1,14), minor=True)
        axis.set(ticks=np.arange(0,14,2)+0.5, ticklabels=np.arange(0,14,2)) #Skipping square labels

    ax.grid(True, which='minor', axis='both', linestyle='--', alpha=0.1, color='k')
    ax.set_xlim((0,13))
    ax.set_ylim((0,13))

    #Colorbar stuff
    cbar = plt.colorbar(heatmap, label="Average response (mV)")
    cbar.ax.get_yaxis().labelpad = 6
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    ax.set_aspect(1)
    fig.set_figheight(8.)
    fig.set_figwidth(8.5)
    simpleaxis(ax,every=True,outward=False)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    # dump(fig,file('figures/fig1/1b.pkl','wb'))
    fig.show()
    
    return list(set(pickedCoords))

def chooseCoordinatesFromOneSquareData(inputDir,maxCoords):
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

    global pickedCoords
    pickedCoords = [] 
    pickedCoords = plotGrid(neuron, maxCoords,pickedCoords = pickedCoords, experimentDir=experimentDir)

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
        self.root.geometry("500x300")
        self.root.wm_title("Coordinate Generator")
        self.plotResponse = 0
        self.maxCoords = 0

        labelframe = Tkinter.LabelFrame(self.root, text="Generate new coordinates ")
        labelframe.pack(fill="both", expand="yes")
        left = Tkinter.Button(labelframe, text = 'Generate randX from scratch', command=self.generateRandXFromScratch)
        left.pack()

        labelframe_2 = Tkinter.LabelFrame(self.root, text="Create randX from CPP values ")
        labelframe_2.pack(fill="both", expand="yes")
        C1 = Tkinter.Checkbutton(labelframe_2, text = "Plot Response", variable = self.plotResponse, onvalue = 1, offvalue = 0, height=5, width = 20)
        right = Tkinter.Button(labelframe_2, text = 'Give location of cell', command=self.generateRandXFromData)
        #right = Tkinter.Button(labelframe_2, text = 'Give location of cell', command=self.generateRandXFromDataByPickingSquares)
        C1.pack()
        right.pack()

        labelframe_3 = Tkinter.LabelFrame(self.root, text="Create randX by choosing grid ")
        labelframe_3.pack(fill="both", expand="yes")

        self.scale = Tkinter.Scale(labelframe_3, orient='horizontal',from_=1,to=30,tickinterval=29, command=self.setMaxCoords)
        #AnswerBox = Tkinter.Entry(labelframe_3)
        #self.maxCoords = int(AnswerBox.get())
        
        #C1 = Tkinter.Checkbutton(labelframe_3, text = "Plot Response", variable = self.maxCoords, onvalue = 1, offvalue = 0, height=5, width = 20)
        #right = Tkinter.Button(labelframe_2, text = 'Give location of cell', command=self.generateRandXFromData)
        right2 = Tkinter.Button(labelframe_3, text = 'Give location of cell', command=self.generateRandXFromDataByPickingSquares)

        #AnswerBox.pack()
        self.scale.pack()
        right2.pack()

        button = Tkinter.Button(self.root, text = 'Close', command=self.quit)
        button.pack()
        self.root.mainloop()

    def setMaxCoords(self, value):
        self.maxCoords = int(value)

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

    def generateRandXFromDataByPickingSquares(self):
        dir = self.get_dir()
        if not dir:
            return
        else:
            print("coordinate", self.maxCoords)
            chooseCoordinatesFromOneSquareData(dir, maxCoords=self.maxCoords)


numEdgeSquares = 13
circleDiameter = 10
skipSquaresBy = 2
diagonalOnly = False
numSquareRepeats = 1

circularGrid = makeCircularPhotostimulationGrid(numEdgeSquares,circleDiameter,skipSquaresBy=skipSquaresBy, diagonalOnly=diagonalOnly)

totalNumberOfSquares = numSquareRepeats * len(circularGrid)

app = App()
