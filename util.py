#!/usr/bin/python2.7

'''
This scripts contains a group of utility functions to allow for
plotting of responses, deconvolution of signal received at the CA1
neuron to synaptic weights at the pre-synaptic stimulation end.
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio 
import itertools as it
from random import sample
from random import shuffle
import scipy
import scipy.stats as ss
import scipy.spatial as sp
import scipy.fftpack as sf
#from sklearn import linear_model
import matplotlib.cm as cm
import sys
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
import glob
import os
from scipy import signal
from scipy.optimize import curve_fit
import copy

colormap = cm.viridis

def randomMatrix(size):
    '''Returns a random matrix with given dims''' 
    nrow,ncol = size
    return np.random.rand(nrow, ncol) 

def randomLogNormalMatrix(size):
    '''Returns a random lognormal matrix with given dims''' 
    lognormal = np.random.lognormal(size=size)
    maxlognormal = np.max(lognormal)
    return lognormal/maxlognormal 

def randomBoolVector(size,numOnes):
    '''Returns a random boolean vector of a given length with a given number of ones''' 
    a = np.zeros(size)
    indexOfOnes = sample(range(len(a)), numOnes)
    a[indexOfOnes] = 1
    return a 

def randomNormalVector(nrow, mu, sigma):
    '''Returns a random normal vector with given mu and sigma''' 
    vec = sigma * np.random.randn(nrow, 1) + mu
    return vec.flatten()

def matrixOfZeros(size):
    ''' Create a matrix with only zeros in it '''
    return np.zeros(size)

def empty_2D_Array(size):
    ''' Create a 2 dimensional array full of empty lists'''
    nrow,ncol = size
    empty_2D_arr = [[[] for j in range(ncol)] for i in range(nrow)]
    return empty_2D_arr

def vectorOfZeros(size):
    return np.zeros(size)

def readMatlabFile(filename):
    '''Reads a matlab file into a dictionary''' 
    fullDict = sio.loadmat(filename)
    for extras in ('__header__','__version__','__globals__'):
        fullDict.pop(extras)
    return fullDict 

def convertDictToList (mat):
    ''' Convert Dict to lists '''
    trace = copy.deepcopy([[mat[key].T[0], mat[key].T[1]] for key in mat.keys()])
    return(trace)

def parseDictKeys(fullDict):
    ''' Converts the dictionary from the matlab file into different dicts of 
    photodiode traces and voltage traces from the CA1'''
    photoDiode = {}
    voltageTrace = {}

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
               totIndex += (j-1)
        i+=1
    return voltageTrace, photoDiode

def readBoxCSV(randx,randy, length= 0):
    ''' Reading the CSV file into list of integers and finally tuples'''
    x = map (int, open(randx).read().splitlines()[0].split(','))
    y = map (int, open(randy).read().splitlines()[0].split(','))

    x = [ i-1 for i in x]
    y = [ i-1 for i in y]

    if not length:
        return x,y
    else:
        return x[:length], y[:length]

def find_BaseLine_and_WindowOfInterest_Margins(photodiode, threshold, baselineWindowWidth, interestWindowWidth):
    ''' Finding the base line windows using this function 
    utilizing the photodiode trace for it.
    '''
    vector = np.array(photodiode[1])
    index = np.where(vector.T[1] > threshold)[0][0]
    baseLineWindow = (index - baselineWindowWidth, index)
    
    interestWindow = (index, index + interestWindowWidth)
    return baseLineWindow, interestWindow

def checkBaseLineStability(volt, baseLineWindow):
    ''' Returns Baseline averaged over window for all CPP trials '''
    ### Has to return a boolean of stability eventually ###
    baseLineStart = []
    for key in volt.keys():
        baseLineStart.append(np.average(volt[key][baseLineWindow[0]:baseLineWindow[1]], axis=0)[1])
    print baseLineStart
    return baseLineStart
    
def normalizeToBaseLine(vector,margins):
    '''normalizes the vector to an average baseline'''
    baseline = vector[margins[0]:margins[1]]
    mu = np.average(baseline)
    newVector = [a - mu for a in vector] 
    return newVector

def normalize (vector):
    ''' Normalizes the vector as to its mean and std '''
    vector = np.array(vector)
    mu = np.average(vector)
    sigma = np.std(vector)
    normalizedVector = (vector-mu)/sigma
    return normalizedVector
    
def findMaximum(vector,margins):
    '''Finds the maximum of the vector in a given window'''
    window = vector[margins[0]:margins[1]]
    max = np.max(window)
    return max

def findMinimum(vector,margins):
    '''Finds the maximum of the vector in a given window'''
    window = vector[margins[0]:margins[1]]
    min = np.min(window)
    return min

def findTimeToPeak(vector, margins, samplingFreq):
    '''Finds the time to maximum of the vector in a given window'''
    window = vector[margins[0]:margins[1]]
    maxIndex = np.argmax(window)
    samplingTime = (1./samplingFreq) # in seconds
    timeToPeak = (maxIndex)*samplingTime
    return timeToPeak 

def findMean(vector,margins):
    '''Finds the mean of the vector in a given window'''
    window = vector[margins[0]:margins[1]]
    mean = np.average(window)
    return mean

def areaUnderTheCurve(vector, margins, samplingFreq):
    '''Finds the area under the curve of the vecotr in the given window'''
    window = vector[margins[0]:margins[1]]
    samplingTime = (1./samplingFreq) # in seconds
    auc = np.trapz(window,dx=samplingTime) # in V.s
    return auc

def areaUnderTheCurveToPeak(vector, margins, samplingFreq):
    '''Finds the area under the curve of the vecotr in the given window'''
    window = vector[margins[0]:margins[1]]
    maxIndex = np.argmax(window)
    samplingTime = (1./samplingFreq) # in seconds
    windowToPeak = window[:maxIndex+1] 
    auctp = np.trapz(windowToPeak,dx=samplingTime) # in V.s
    return auctp

def smoothenVector(vector,smootheningWindow):
    ''' This uses a moving vector average to smoothen out
    the vector of interest.'''
    window= np.ones(int(smootheningWindow))/float(smootheningWindow)
    return np.convolve(vector, window, 'same')

def addTraces(voltageTrace, coords, numSquares):
    ''' Adds voltage traces to each other for single square stimulation, to mimic output when stimulation has happened together on these boxes'''
    summedInput = {}
    x,y = coords
    maxKey = len(voltageTrace.keys())+1
    xy = it.cycle(it.izip(x,y)) # This will keep doing an indefinite cycling through the coordinates.
    key = 1
    newkey = 1
    while key < maxKey:
        fiveSquareSum = vectorOfZeros( len(voltageTrace[key].T[1])) 
        for i in range(numSquares): 
            fiveSquareSum = np.add(fiveSquareSum, voltageTrace[key].T[1])
            key+=1
            if key >= maxKey:
                break
        summedInput.update({newkey: [voltageTrace[1].T[0],fiveSquareSum.T]})
        newkey+=1
    return summedInput

def alphaFunction(t,tau):
    ''' Returns the shape of an EPSP as an alpha function '''
    g = (t/tau)*np.exp(1-t/tau)
    return g

def doubleExponentialFunction(t, tOn,tOff):
    ''' Returns the shape of an EPSP as a double exponential function '''
    tPeak = float(((tOff * tOn)/(tOff-tOn)) * np.log(tOff/tOn))
    A = 1./(np.exp(-tPeak/tOff) - np.exp(-tPeak/tOn))
    g = A * ( np.exp(-t/tOff) - np.exp(-t/tOn))
    return g

def fitDoubleExpToEPSP(vector, margins, samplingFreq):
    window = vector[margins[0]:margins[1]]
    samplingTime = (1./samplingFreq) # in seconds
    time = np.arange(len(window))*samplingTime
    popt,pcov = curve_fit(doubleExponentialFunction,time, window) 
    return popt, pcov

def mapOutputToRandomStimulation(outputVarDict, coords, randStimBox, numSquares):
    ''' Map output voltage traces to the random stimulation
    box in terms of repeats of squares. This can also be used to assign the contributions
    in a weighted manner, as opposed to uniform, by passing the contribution matrix in the randStimBox.
    '''
    x,y = coords
    keyrange = range(1, len(outputVarDict.keys())+1)
    xy = it.cycle(it.izip(x,y)) # This will keep doing an indefinite cycling through the coordinates.

    if (np.array(randStimBox).size):
        outBox = empty_2D_Array(np.array(randStimBox).shape)

        for key in keyrange:
           cumProb = []
           tempCoord = []
           for i in range(numSquares):
               xValue, yValue = xy.next()
               cumProb.append(randStimBox[xValue][yValue])
               tempCoord.append((xValue,yValue))
           nonZeroSum = np.sum(cumProb)
           if (nonZeroSum):
               prob = np.divide(cumProb, nonZeroSum )
           else:
               prob = cumProb

           for i in range(len(tempCoord)):
               outBox[tempCoord[i][0]][tempCoord[i][1]].append( outputVarDict[key] * prob[i]) 
        return outBox
    ########## If the randStimBox is empty, uniform distribution of contribution is assumed.
    else:
        uniformProb = 1./numSquares
        for key in keyrange:
            for i in range(numSquares):
                xValue, yValue = xy.next()
                if key in outputVarDict:  ## Making provision for empty trials.
                    randStimBox[xValue][yValue].append(outputVarDict[key] * uniformProb )
                else:
                    randStimBox[xValue][yValue]  = [None]
        return randStimBox

def mapOutputToInputCombinations(outputVarDict, coords, numSquares, AP_flag, baseline_flag, noise_flag, photodiode_flag):
    ''' Map output voltage traces to combinations of input squares 
    box in terms of repeats of squares. This will help average out combinations from the patterned inputs. 
    '''
    x,y = coords
    keyrange = range(1, len(outputVarDict.keys())+1)
    xy = it.cycle(it.izip(x,y)) # This will keep doing an indefinite cycling through the coordinates.
    outputToInputCombs = {}
    newMappingValue = {key: value for key, value in outputVarDict.items()}
    for key in keyrange:
        currentCoordinate = []
        for i in range(numSquares):
            currentCoordinate.append(xy.next())
        currentCoordinate = tuple(currentCoordinate)

        # Ignoring trials with these problems
        if AP_flag[key] == 1 or baseline_flag == 1 or photodiode_flag ==1:
            print "AP", key
            newMappingValue.pop(key)
            continue
        print key
        # Setting noisy trials to zero. 
        if currentCoordinate not in outputToInputCombs:
            if noise_flag[key] == 0:
                outputToInputCombs[currentCoordinate] = [outputVarDict[key]]
            else:
                outputToInputCombs[currentCoordinate] = 0. 
        else:
            if noise_flag[key] == 0:
                outputToInputCombs[currentCoordinate] = [outputVarDict[key]]
            else:
                outputToInputCombs[currentCoordinate] = 0. 

    meanOutputToInputCombs = {}

    for key in outputToInputCombs.keys():
        meanOutputToInputCombs[key] = np.mean(outputToInputCombs[key])
    return newMappingValue, outputToInputCombs, meanOutputToInputCombs 

def addRandomStimulationResponses(coords, randStimBox, numSquares, totalResponses):
    ''' Map back from the random stimulation box to cumulative output voltage by adding up squares. 
    '''
    x,y = coords
    xy = it.cycle(it.izip(x,y)) # This will keep doing an indefinite cycling through the coordinates.
    outputVec = []
    outputVecDict={}
    for j in range(totalResponses):
       summedResponse= 0.
       coordNow = []
       for i in range(numSquares):
           xValue, yValue = xy.next()
           coordNow.append((xValue, yValue))
           summedResponse += randStimBox[xValue][yValue]
       outputVec.append(summedResponse)
       outputVecDict[tuple(coordNow)] = summedResponse
    return outputVec, outputVecDict

def mappedMatrixStatistics(coords, randStimBox):
    ''' This does statistics on the output variable matrix mapped 
    to the random stimulation'''

    randStimBoxSize = (len(randStimBox),len(randStimBox[0]))
    x,y = coords
    randStimBoxMean = matrixOfZeros(randStimBoxSize)
    randStimBoxNumStims = matrixOfZeros(randStimBoxSize)
    randStimBoxVar = matrixOfZeros(randStimBoxSize)

    for xy in zip (x,y):
        if (randStimBox[xy[0]][xy[1]]):
            randStimBoxNumStims[xy] = len(randStimBox[xy[0]][xy[1]])
            randStimBoxMean[xy] = np.average(randStimBox[xy[0]][xy[1]])
            if randStimBoxMean[xy]!=0:
                randStimBoxVar[xy] = np.var(randStimBox[xy[0]][xy[1]]) / randStimBoxMean[xy]
            else:
                randStimBoxVar[xy] = np.nan

    return randStimBoxNumStims, randStimBoxMean, randStimBoxVar

def convert_1dim_DictToList (mat):
    ''' Convert Dict to lists '''
    keyList = sorted(mat.keys())
    trace = [mat[key] for key in keyList]
    return(trace)

def plotScatter_2_dicts(dict1, dict2, labels = ('x','y'), title = 'Untitled', showPlots = False, plotType = 'png', outFile = 'scatterPlot'):
    ''' Prepares a scatter plot from two dictionaries with the same keys '''
    x = []
    y = []
    for key in dict1.keys():
        x.append(dict1[key])
        y.append(dict2[key])

    plt.scatter(x,y)
    plt.title (title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.xlim((0.8*min(x), 1.2*max(x)))
    plt.ylim((0.8*min(y), 1.2*max(y)))
    plt.tight_layout()
    if (showPlots):
        plt.show()
    else:
        plt.savefig(outFile + '.' + plotType)
    plt.close()

def plotScatterWithRegression(xdata,ydata,SizeOfPhotoactiveGrid,colorby=None, xlabel='',ylabel='',title='', axis=None):
    ############ Plotting scatter plots for comparison ##############
    #xmin, xmax = axis.get_xlim()
    axisWidth = (1.05*min(min(xdata),min(ydata)),1.05*max(max(xdata),max(ydata)))
    axis.set(adjustable='box-forced', aspect='equal')
    axis.set_xlim(axisWidth)
    axis.set_ylim(axisWidth)
    axis.scatter(xdata,ydata, s=20, facecolor='none', edgecolor=colorby, lw='1.0',alpha=0.6)
    axis.plot(axisWidth, axisWidth, 'r--' )
    axis.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    #axis.annotate('Supra-linear', xy=(0.70*axisWidth[1], 0.70*axisWidth[1] ), xytext=(0.60*axisWidth[1],0.80*axisWidth[1]), arrowprops=dict(arrowstyle='<-'), ha='center', va='center')
    #axis.annotate('Sub-linear', xy=(0.70*axisWidth[1],0.70*axisWidth[1] ), xytext=(0.80*axisWidth[1], 0.60*axisWidth[1]), arrowprops=dict(arrowstyle='<-'), ha='center', va='center')
    #plt.title(measureList[measure])
   
    #axis.xaxis.offsetText.set_visible(False)
    #axis.yaxis.offsetText.set_visible(False)

    axis.set_title(title, size='x-small')
    axis.tick_params(axis='both', labelsize=8)
   
    #fitting a regression
    fitParameters = fitLinearRegressor(xdata, ydata, domain=SizeOfPhotoactiveGrid, axis=axis,color=colorby)
    return fitParameters
 
def createArtificialData(numTrials, synapticWeightMatrix, projection, synapseParameters, numStimulatedSquares = 1, mu_pre = 0.003, sigma_pre = 0.0005, mu_post = 0.0, sigma_post = 0.0, gridShape= (15,15)):

    ''' This function creates artificial photostimulation data
    M = N * (alpha*P)*epsp + beta ## The main equation 
    '''
    signal = [] # Measurement from post-synaptic neurons
    N = synapticWeightMatrix # Synaptic Weight matrix (pre-synaptic grid shape)
    flatN = N.flatten()
    totalGridSize = len(flatN)

    responseDuration = 1000 
    totalSignalDuration = 5000 
    signalStart = 2000  ## Can also be manipulated

    frameRate = 10. ## kHz
    maxT = responseDuration/frameRate ## in ms, Can also be manipulated
    t = np.linspace(0,maxT, frameRate*maxT)

    if len(synapseParameters) == 1:
        ##### For alpha function synapses ####################
        tau = synapseParameters[0]  ## in ms, Can also be manipulated
        epsp = alphaFunction(t,tau)
    elif len(synapseParameters) == 2:
        ##### For double exponential function synapses ########
        tOn = synapseParameters[0]  ## in ms 
        tOff= synapseParameters[1] 
        epsp = doubleExponentialFunction(t, tOn,tOff)
    else:
        print "wrong number of parameters entered"
        sys.exit() 

    for trial in range(numTrials):
        alpha = randomNormalVector(totalGridSize, mu_pre, sigma_pre) # Channelrhodoposin distribution, pre-synaptic release probability.
        P = projection[trial]
        measure = np.dot( np.multiply(alpha,flatN),P)
        sig = measure * epsp 

        baseLine = np.zeros(totalSignalDuration)
        baseLine[ signalStart : signalStart + responseDuration ] = sig 
        beta = randomNormalVector(totalSignalDuration, mu_post, sigma_post) # Post-synaptic response variability, measurement noise. 
        signal.append(baseLine*(1. + beta))

    return signal 

def createArtificial_RS_Trace(numInputs, synapseParameters, mu_amp = 0.003, sigma_amp = 0.0005, mu_t = 0.1, sigma_t = 0.05, acquisitionRate=20000):

    ''' This function creates artificial photostimulation data
    M = N * epsp  ## The main equation 
    '''
    signal = [] # Measurement from post-synaptic neurons

    meanIndex = int(mu_t*acquisitionRate )
    deltaIndices = np.random.poisson(meanIndex, numInputs) 
    deltaT = deltaIndices/acquisitionRate
    
    timeIndices = np.cumsum(deltaIndices ) # Time calculated by cumulative summing on an axis 
    time = timeIndices/acquisitionRate
    t=10*mu_t # Because epsp curves asymptote to zero.

    if len(synapseParameters) == 1:
        ##### For alpha function synapses ####################
        tau = synapseParameters[0]  ## in ms, Can also be manipulated
        epsp = alphaFunction(np.linspace(0.,t,t*acquisitionRate),tau)
    elif len(synapseParameters) == 2:
        ##### For double exponential function synapses ########
        tOn,tOff = synapseParameters  ## in ms 
        epsp = doubleExponentialFunction(np.linspace(0,t,t*acquisitionRate), tOn,tOff)
    else:
        print "wrong number of parameters entered"
        sys.exit() 

    #alpha= np.random.poisson(mu_amp, numInputs) 
    alpha = randomNormalVector(numInputs, mu_amp, sigma_amp) # Magnitude of EPSP variability.
    sig = [np.array(epsp)*a for a in alpha]
    #for s in sig:
    #    plt.plot(range(len(s)), s)
    #plt.show()
    baseLine = np.zeros(1.1*time[-1]*acquisitionRate)
    
    for index,signal in zip(timeIndices,sig):
        baseLine[ index : index + len(signal) ] += signal 
    return baseLine 


def syntheticNetworkData(CA3_Neurons = 15, CA1_Neurons = 1, mu_pre = 2., sigma_pre = 0.6, mu_post = 1., sigma_post = 0.3):

    ''' This function will try to deconvolve a synaptic weight matrix from its post-synaptic measurements.'''

    numNeurons = CA3_Neurons + CA1_Neurons
    
    alpha = randomNormalVector(numNeurons, mu_pre, sigma_pre) # Channelrhodoposin distribution, pre-synaptic release probability.
    beta = randomNormalVector(numNeurons, mu_post, sigma_post) # Post-synaptic response variability, measurement noise. 
    
    M = np.zeros((numNeurons,1)) # Measurement from post-synaptic neurons
    N = randomMatrix((numNeurons,numNeurons)) # Synaptic Weight matrix (post-synaptic x pre-synaptic)
    P = randomBoolVector(numNeurons) # Projection matrix (light)

    # N * (alpha*P) = beta * M ## The main equation 
    M = np.multiply(1/beta , np.dot(N, np.multiply(alpha,P))) ## The main equation 
    #plt.hist(M,bins=1000)
    #plt.show()
    return alpha, beta, M, N, P

#def findMaxDict(voltageTrace,marginOfBaseLine,marginOfInterest, smootheningWindow, filtering='', Low_cutoff = 0, High_cutoff=1000, F_sample= 20000, leastCount = 0.0005 ):
#    ''' Creating a dict with the maximum values for each key, fourier transform and smoothen the curve, and threshold with least count of the measurement.'''
#    maxValue = {}
#    AP_threshold = 9e-2 
#    for key in voltageTrace.keys():
#        #plt.plot(voltageTrace[key].T[0], voltageTrace[key].T[1])
#        if (filtering=='ifft_bandpass'):
#            Spectrum, Filtered_spectrum, Filtered_signal, Low_point, High_point = bandpass_ifft(voltageTrace[key].T[1], Low_cutoff, High_cutoff, F_sample)
#            voltageTrace[key].T[1] = Filtered_signal
#        elif (filtering=='bessel'):
#            Filtered_signal = besselFilter(voltageTrace[key].T[1], cutoff=High_cutoff, F_sample=F_sample)
#            voltageTrace[key].T[1] = Filtered_signal
#
#        voltageTrace[key].T[1] = normalizeToBaseLine(voltageTrace[key].T[1],marginOfBaseLine)
#        if np.max(voltageTrace[key].T[1]) > AP_threshold:
#            print "Action Potential with parameters: key:{}, amplitude:{}".format(key, np.max(voltageTrace[key].T[1]) - np.min(voltageTrace[key].T[1])) 
#            #plt.plot(voltageTrace[key].T[0], voltageTrace[key].T[1])
#            #plt.show()
#        voltageTrace[key].T[1] = smoothenVector(voltageTrace[key].T[1], smootheningWindow)
#        mappingValue = findMaximum(voltageTrace[key].T[1],marginOfInterest)
#        ##### Thresholding measure by virtue of measurement least count ##########
#        #mappingValue = mappingValue if mappingValue > leastCount else 0. 
#        maxValue.update({key :mappingValue }) 
#
#    return maxValue

def flagActionPotentials(trace, marginOfInterest, AP_threshold=3e-2):
    ''' This function flags if there is an AP trialwise and returns a dict of bools '''
    if np.max(trace[marginOfInterest[0]:marginOfInterest[1]]) > AP_threshold:
        return 1 
    else:
        return 0

def flagBaseLineInstability(voltageTrace, baseline):
    ''' This function flags if the baseline is too variable for measuring against '''
    pass

def flagPhotodiodeInstability(photodiode, margin):
    ''' This function flags if the photodiode trace is too noisy '''
    pass


def flagNoise(trace, baseline, interest, pValTolerance = 0.05):
    ''' This function asseses if the distributions of the baseline and interest are different or not '''
    m, pVal = ss.ks_2samp(trace[baseline[0]:baseline[1]],trace[interest[0]:interest[1]])
    if pVal < pValTolerance:
        return 0
    else:
        return 1 # Flagged as noisy


def findMeasureDict(measure, voltageTrace,marginOfBaseLine,marginOfInterest, smootheningWindow, filtering = '', Low_cutoff = 0, High_cutoff=1000, F_sample= 20000, leastCount = 0.0005 ):
    ''' Creating a dict with the area under the curve values for each key'''
    mappingDict = {}
    AP_flag = {}
    noise_flag = {}
    baseline_flag = {}
     
    for key in voltageTrace.keys():
        #plt.plot(voltageTrace[key].T[0], voltageTrace[key].T[1])
        if (filtering=='ifft_bandpass'):
            Spectrum, Filtered_spectrum, Filtered_signal, Low_point, High_point = bandpass_ifft(voltageTrace[key].T[1], Low_cutoff, High_cutoff, F_sample)
            voltageTrace[key].T[1] = Filtered_signal
        #plt.plot(voltageTrace[key].T[0], voltageTrace[key].T[1])
        #plt.show()
        elif (filtering=='bessel'):
            Filtered_signal = besselFilter(voltageTrace[key].T[1], cutoff=High_cutoff, F_sample=F_sample)
            voltageTrace[key].T[1] = Filtered_signal

        baseline_flag.update({key : flagBaseLineInstability(voltageTrace[key].T[1], marginOfBaseLine)})
        voltageTrace[key].T[1] = normalizeToBaseLine(voltageTrace[key].T[1],marginOfBaseLine)
        AP_flag.update({key: flagActionPotentials(voltageTrace[key].T[1], marginOfInterest)})
        noise_flag.update({key: flagNoise(voltageTrace[key].T[1], marginOfBaseLine, marginOfInterest)})
         
        voltageTrace[key].T[1] = smoothenVector(voltageTrace[key].T[1], smootheningWindow)
        if measure == 0:
            mappingValue = findMaximum(voltageTrace[key].T[1],marginOfInterest)
        elif measure == 1:
            mappingValue = areaUnderTheCurve(voltageTrace[key].T[1],marginOfInterest, F_sample)
        elif measure == 2:
            mappingValue = findMean(voltageTrace[key].T[1],marginOfInterest)
        ##### Thresholding measure by virtue of measurement least count ##########
        #leastCountArea = areaUnderTheCurve( np.repeat(leastCount, len(voltageTrace[key].T[1])), marginOfInterest, F_sample)
        #mappingValue = mappingValue if mappingValue > leastCountArea else 0. 
        elif measure == 3:
            mappingValue = findTimeToPeak(voltageTrace[key].T[1],marginOfInterest, F_sample)
        elif measure == 4:
            mappingValue = areaUnderTheCurveToPeak(voltageTrace[key].T[1],marginOfInterest, F_sample)
        elif measure == 5:
            mappingValue = findMinimum(voltageTrace[key].T[1],marginOfInterest)
 
        mappingDict.update({key :mappingValue }) 
    return mappingDict, AP_flag, baseline_flag, noise_flag

def acorr(x, ax=None):
    if ax is None:
        ax = plt.gca()

    x = x - x.mean()

    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[x.size:]
    autocorr /= autocorr.max()

    return ax.stem(autocorr)

def mapOutputToGrid(recording, coords, SizeOfPhotoactiveGrid, measure=0, weightDist = [], samplingTime = 0.05, threshold = 0.05, smootheningTime = 0.05, baseline= 100., interest= 50., randStimBoxSize = (13,13), randomizeTrials = False, filtering=''):
    #### Creating the Random stimulation box and other callibration values here
    smootheningWindow = int(smootheningTime/samplingTime )
    baselineWindowWidth = int(baseline/samplingTime)
    interestWindowWidth = int(interest/samplingTime)

    #### Reading the voltage and photodiode traces here #####
    fullDict = readMatlabFile(recording)
    voltageTrace, photoDiode = parseDictKeys(fullDict)
    
    #  For selecting a few trials only for measure
    #for key in voltageTrace.keys():
    #    if (SizeOfPhotoactiveGrid == 5 and key > 90):
    #        voltageTrace.pop(key)

    #  For selecting a few trials only for measure
    #for key in voltageTrace.keys():
    #    if (SizeOfPhotoactiveGrid == 1 and 167<key<179):
    #        np.transpose(voltageTrace[key])[1] = np.zeros(len(np.transpose(voltageTrace[key])[1]))

    #### Find the baselines and window of interest from the photodiode trace
    marginOfBaseLine, marginOfInterest = find_BaseLine_and_WindowOfInterest_Margins(photoDiode,threshold, baselineWindowWidth, interestWindowWidth)

    ### Creating a dict with fourier bandpassed and smoothened output, where output variable = maximum amplitude of voltage traces ###
    mappingValue, AP_flag, baseline_flag, noise_flag = findMeasureDict( measure, voltageTrace,marginOfBaseLine,marginOfInterest, smootheningWindow, filtering=filtering)
    photodiode_flag = flagPhotodiodeInstability(photoDiode, marginOfInterest)
    #print AP_flag
    #print "baseline_flag"
    #print baseline_flag
    #print "noise_flag"
    #print noise_flag

    ######## Randomizing the keys and values of the mappingValue ##########
    if randomizeTrials:
        keys = mappingValue.keys()
        values = mappingValue.values()
        shuffle(values)
        mappingValue = dict(zip(keys, values))
    ### Mapping this dict onto the random stimulation box here, either uniformly or with a weight ####
    if not np.any(weightDist):
        randStimBox = empty_2D_Array( randStimBoxSize)
    else:
        randStimBox = weightDist

    clusterDistance = mapSpatialCorrelation(mappingValue, coords, SizeOfPhotoactiveGrid, randStimBoxSize= randStimBoxSize )
    randStimBox = mapOutputToRandomStimulation(mappingValue, coords, randStimBox, SizeOfPhotoactiveGrid)
    newMappingValue, outputToInputCombs, meanOutputToInputCombs = mapOutputToInputCombinations(mappingValue, coords, SizeOfPhotoactiveGrid, AP_flag, baseline_flag, noise_flag, photodiode_flag)
    randStimBoxNumStims, randStimBoxMean, randStimBoxVar = mappedMatrixStatistics(coords, randStimBox)
    return newMappingValue, outputToInputCombs,meanOutputToInputCombs, randStimBoxMean, randStimBoxVar, randStimBox, clusterDistance

def mapSpatialCorrelation(outputVarDict, coords, numSquares, randStimBoxSize):
    ''' Finds the correlation between photostimulation squares and the output from CA1'''

    x,y = coords
    keyrange = range(1, len(outputVarDict.keys())+1)
    clusterDistance = dict.fromkeys(outputVarDict.keys())
    xy = it.cycle(it.izip(x,y)) # This will keep doing an indefinite cycling through the coordinates.

    for key in keyrange:
       tempCoord = []
       for i in range(numSquares):
           xValue, yValue = xy.next()
           tempCoord.append((xValue,yValue))
       clusterDistance[key] =  findNormalizedTotalDistance(tempCoord, normalize = distance((0,0), (randStimBoxSize[0]-1,randStimBoxSize[1]-1)))
    return clusterDistance 

def findNormalizedTotalDistance(listname, normalize =1.):
    distanceVal = 0.
    totalLength = 1.
    for i in range(len(listname)-1):
        for j in range(i+1,len(listname)):
            distanceVal += distance(listname[i],listname[j])/normalize
            totalLength+=1
    if totalLength !=1:
        totalLength -= 1 # Factorials because of the loops
    return distanceVal/totalLength

def bandpass_ifft(X, Low_cutoff, High_cutoff, F_sample, nfactor=1):
        """Bandpass filtering on a real signal using inverse FFT
        Inputs
        =======
        X: 1-D numpy array of floats, the real time domain signal (time series) to be filtered
        Low_cutoff: float, frequency components below this frequency will not pass the filter (physical frequency in unit of Hz)
        High_cutoff: float, frequency components above this frequency will not pass the filter (physical frequency in unit of Hz)
        F_sample: float, the sampling frequency of the signal (physical frequency in unit of Hz)    
        """        
        
        M = X.size # let M be the length of the time series
        Spectrum = sf.rfft(X, n=M) 
        [Low_cutoff, High_cutoff, F_sample] = map(float, [Low_cutoff, High_cutoff, F_sample])
        
        #Convert cutoff frequencies into points on spectrum
        [Low_point, High_point] = map(lambda F: F/F_sample * M, [Low_cutoff, High_cutoff])
        Filtered_spectrum = [Spectrum[i] if i >= Low_point and i <= High_point else 0.0 for i in xrange(M)] # Filtering
        N = M / nfactor
        Filtered_signal = sf.irfft(Filtered_spectrum, n=N)  # Construct filtered signal 
        return Spectrum, Filtered_spectrum, Filtered_signal, Low_point, High_point

def besselFilter(X, order=2, cutoff=150., F_sample=20000.):
    ''' Applies a bessel filter to the signal.'''
    #cutoff_to_niquist_ratio = cutoff/(np.pi*2) # F_sample/2 is Niquist, cutoff is the low pass cutoff.
    cutoff_to_niquist_ratio = 2*cutoff/(F_sample) # F_sample/2 is Niquist, cutoff is the low pass cutoff.
    b, a = signal.bessel(order, cutoff_to_niquist_ratio, btype='low', analog=False)
    output_X = scipy.signal.filtfilt(b, a, X)
    return output_X

def plotTrace(traces, marginOfBaseLine, interestWindow, outFile=[]):
    '''Just plots traces for a list with the time and voltage lists''' 
    f,ax = plt.subplots()
    for trace in traces:
        t = trace[0]
        v = normalizeToBaseLine(trace[1],marginOfBaseLine)
        ax.plot(t,v,'-')
    ylim = ax.get_ylim()
    ax.axvspan(t[marginOfBaseLine[0]], t[marginOfBaseLine[1]], facecolor='b', alpha=0.25)
    ax.text(1.05*t[marginOfBaseLine[0]], 1.05*ylim[1], 'Baseline', fontsize=12)
    ax.axvspan(t[interestWindow[0]], t[interestWindow[1]], facecolor='g', alpha=0.25)
    ax.text(1.05*t[interestWindow[0]], 1.05**ylim[1], 'Interest', fontsize=12)
    plt.tight_layout()
    if not outFile:
        plt.show()
    else:
        plt.savefig(outFile)
    plt.close()

def plotHeatMapBox(matrix, title, measure, outputDir='./', showPlots=1, filename = [], plotGridOn =[]):
    ''' Plots the heatmap of the output variable 
    in the randomStimulation box'''
    
    nrow,ncol = matrix.shape 
    alpha = 1
    column_labels = [str(i+1) for i in range(ncol)]
    row_labels = [str(i+1) for i in range(nrow)]
    
    fig, ax = plt.subplots()
    
    mask = np.ma.make_mask(matrix)
    masked_matrix = np.ma.masked_where(~mask,matrix)
    cmap=colormap
    cmap.set_bad('w',1.)

    if (plotGridOn):
        alpha = 0.25
        im = plt.imread(plotGridOn)
        implot = plt.imshow(im, extent=[1, nrow-1, +1, ncol-1], cmap = cm.binary)
        implot.set_interpolation('nearest')

    fmt = ScalarFormatter()
    fmt.set_powerlimits((0,0))  

    plt.gca().set_xlim((0,ncol))
    plt.gca().set_ylim((0,nrow))

    heatmap = ax.pcolor( masked_matrix, vmin= matrix.min(), vmax = matrix.max(), alpha = alpha, cmap=cmap)
    #heatmap = ax.pcolor( masked_matrix, vmin= -0.000025, vmax = 0.0002, alpha = alpha)

    cbar = plt.colorbar(heatmap, format=fmt, shrink=0.95, pad = 0.02)
    cbar.ax.tick_params(labelsize=15) 
    cbar.ax.yaxis.get_offset_text().set_position((3.0,1))
    cbar.set_label(r'${}$'.format(measure), y=0.45)
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(nrow)+0.5, minor=False)
    ax.set_yticks(np.arange(ncol)+0.5, minor=False)
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False, size='small')
    ax.set_yticklabels(column_labels, minor=False, size='small')
    ax.set_title(title, y = 1.06, size='medium')
    #plt.tight_layout()

    if not showPlots:
        if (filename):
            plt.savefig(outputDir + measure + '_' + filename)
        else:
            plt.savefig(outputDir + measure + '_' + 'heatMap.png')
    else:
        plt.show()

    plt.close()

def plotContourMap(matrix, title):
    x= np.arange(1,16,1)
    y= np.arange(1,16,1)

    X,Y = np.meshgrid(x, y)
    plt.axes([0.025, 0.025, 0.95, 0.95])
    plt.contourf(X, Y, matrix, 8, alpha=.75, cmap=colormap)
    C = plt.contour(X, Y, matrix, 8, colors='black', linewidth=.5)
    plt.clabel(C, inline=1, fontsize=10)

    plt.xticks(())
    plt.yticks(())
    plt.show()

def plotHistogramFromList(listname, title, outputDir='./', filename = [], bins=[], labels = [], showPlots=1): 
    ''' Plot a histogram with the given parameters'''
    if not bins:
        bins = len(listname)/5
    plt.hist(listname,bins=bins)
    plt.title (title , y = 1.02)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.setp(plt.gca().get_xticklabels(), fontsize=10)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if not showPlots:
        if (filename):
            plt.savefig(outputDir + filename)
        else:
            plt.savefig(outputDir + 'histogram.png')
    else:
        plt.show()
    plt.close()

def plotHistogram(matrix,title):
    list = matrix.flatten().tolist()
    plt.hist(list,bins=len(list)/5)
    plt.title(title)
    #style='sci'
    plt.xlabel('Maximum amplitude of voltage trace in the window of interest')
    plt.ylabel('Frequency')
    plt.show()

def fitDistributionAndPlot(variable, indices = []):
    ''' Fit Distribution Data '''
    dist_names = [ 'alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponweib', 'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic', 'genpareto', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'logistic', 'loggamma', 'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh', 'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform', 'vonmises', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy']
   
    if not indices:
        indices = range(len(dist_names))
    dist_names = [dist_names[index] for index in indices]
    maxVariable = max(variable)
    minVariable = min(variable)

    size, steps = len(variable), 1e-2*(maxVariable - minVariable)
    bins = scipy.arange(minVariable, maxVariable, steps)
    plt.hist(variable, bins = bins, histtype = 'step')
    
    for dist_name in dist_names:
        dist = getattr(ss, dist_name)
        param = dist.fit(variable)
        pdf_fitted = dist.pdf(bins, *param[:-2], loc=param[-2], scale=param[-1]) *steps *size
        plt.plot(bins, pdf_fitted, label=dist_name, linestyle='-')
        plt.xlim(minVariable,maxVariable)
    plt.legend(loc='upper right', fontsize = 'xx-small')
    plt.show()

#def fitLinearRegressor(xdata, ydata):
    #xdata,ydata = np.array([xdata]).T, np.array(ydata)
    #print xdata.shape, ydata.shape

    ## Fit line using all data
    #model = linear_model.LinearRegression()
    #model.fit(xdata, ydata)

    ## Robustly fit linear model with RANSAC algorithm
    ##model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    ##model_ransac.fit(xdata, ydata)
    ##inlier_mask = model_ransac.inlier_mask_
    ##outlier_mask = np.logical_not(inlier_mask)

    ## Predict data of estimated models
    ##axisWidth = (1.05*min(min(x),min(y)),1.05*max(max(x),max(y)))
    ##line_X = np.arange(min(min(xdata),min(ydata)), max(max(xdata),max(ydata)))
    ##line_X = np.arange(1.05*min(min(xdata),min(ydata)),1.05*max(max(xdata),max(ydata)))
    #line_X = xdata.flatten() 
    #line_y = model.predict(line_X[:, np.newaxis])
    ##score = model.score(line_y.flatten(), ydata.flatten())
    ##line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])

    ## Compare estimated coefficients
    ##print("Estimated coefficients (normal, RANSAC):")
    ##print(model.coef_, model_ransac.estimator_.coef_)

    ##plt.plot(xdata[inlier_mask], ydata[inlier_mask], '.g', label='Inliers')
    ##plt.plot(xdata[outlier_mask], ydata[outlier_mask], '.r', label='Outliers')
    #plt.plot(line_X, line_y, '-k', label='Linear regressor, $R^2$ = '+ str(score))
    ##plt.plot(line_X, line_y_ransac, '-b', label='RANSAC regressor')
    #plt.legend(loc='lower right')

def fitLinearRegressor(xdata, ydata, domain='',axis=[],color=''):
    if not len(color):
        colorVec = cm.viridis(np.linspace(0, 1, 16)) # Assuming 16 is never crossed in photostimulation. 
        color = colorVec[domain]
    slope, intercept, r_value, p_value, std_err = ss.linregress(xdata,ydata)
    ynew = slope*np.array(xdata) + intercept
    if axis:
        axis.plot(xdata, ynew, color=color, label = str(domain) + ' square ' + '{:.2f}*x + {:.2f}, $R^2$ ={:.2f}'.format(slope,intercept,r_value))
    else:
        plt.plot(xdata, ynew, '--', color=color, label = '{:.2f}*x + {:.2f}, $R^2$ ={:.2f}'.format(slope,intercept,r_value))
        plt.legend(loc='upper left', fontsize='small')
    return np.array([slope, intercept]) 

def writeToFile(data, filename):
    np.savetxt(filename, data)

def distance(a,b, measure='euclidean'):
    if measure == 'euclidean':
        return sp.distance.euclidean(a,b) 
    elif measure == 'correlation':
        return sp.distance.correlation(a,b) 
    elif measure == 'ks2samp':
        return ss.ks_2samp(a,b)
    elif measure == 'mannwhitneyu':
        return ss.mannwhitneyu(a,b)
    elif measure == 'wilcoxon':
        return ss.wilcoxon(a,b)
    elif measure == 't-test':
        return ss.ttest_rel(a,b)

    else:
        raise "Unknown measure"

def getInputSizeOfPhotoActiveGrid(inputDir):
    photoActiveSquaresList = []
    for cppFile in glob.glob(inputDir + 'CPP?.mat'):
        photoActiveSquaresList.append(int(cppFile.split('.')[0][-1]))
    return sorted(photoActiveSquaresList)

def createCoords(randX_file, randY_file, repeatSize, SizeOfPhotoactiveGrid, inputDir):
    ''' Creates photostimulation coords for the CPP dateset from the randcoords '''

    if not (os.path.isfile(inputDir + 'coords/CPP' + str(SizeOfPhotoactiveGrid) + '_randX.txt') or os.path.isfile(inputDir + 'coords/CPP' + str(SizeOfPhotoactiveGrid) + '_randY.txt')):
        randX = np.loadtxt(randX_file,delimiter=',')
        randY = np.loadtxt(randY_file,delimiter=',')

        print "Repeat is", repeatSize, SizeOfPhotoactiveGrid

        randX_sizeBased = randX[:(repeatSize*SizeOfPhotoactiveGrid)]
        randY_sizeBased = randY[:(repeatSize*SizeOfPhotoactiveGrid)]

        randX_newFile = inputDir + 'coords/CPP' + str(SizeOfPhotoactiveGrid) + '_randX.txt'
        randY_newFile = inputDir + 'coords/CPP' + str(SizeOfPhotoactiveGrid) + '_randY.txt'
        np.savetxt(randX_newFile,randX_sizeBased[None],fmt='%d',delimiter=',')
        np.savetxt(randY_newFile,randY_sizeBased[None],fmt='%d',delimiter=',')
