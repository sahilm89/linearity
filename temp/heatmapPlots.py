#!/usr/bin/python2.7

'''
This script contains the code to plot heatmaps from the 
response neuron based on random stimulation, and do statistical 
analysis on them.
'''
from util import *
import matplotlib.pyplot as plt

def main():
    
    #### Input files here ######################################
    recording = '/home/sahil/Documents/Codes/bgstimPlasticity/data/CPP.mat'
    randX = '/home/sahil/Documents/Codes/bgstimPlasticity/coords/randX.txt'
    randY = '/home/sahil/Documents/Codes/bgstimPlasticity/coords/randY.txt'
   
    #### Creating the Random stimulation box and other callibration values here
    samplingTime = 0.05 # ms
    randStimBoxSize = (15,15)
    SizeOfPhotoactiveGrid = 1 # Number of squares on the grid which are photostimulated

    threshold = 0.05  # 50 mV 
    smootheningTime = 0.05 # ms
    baseline= 100. # ms
    interest= 50. # ms

    smootheningWindow = int(smootheningTime/samplingTime )
    baselineWindowWidth = int(baseline/samplingTime)
    interestWindowWidth = int(interest/samplingTime)

    #### Reading the voltage and photodiode traces here #####
    fullDict = readMatlabFile(recording)
    voltageTrace, photoDiode = parseDictKeys(fullDict)
    
    #### Find the baselines and window of interest from the photodiode trace
    marginOfBaseLine, marginOfInterest = find_BaseLine_and_WindowOfInterest_Margins(photoDiode,threshold, baselineWindowWidth, interestWindowWidth)

    ### Reading random stimulation coordinates from file here ###
    coords = readBoxCSV(randX,randY)

    ### Creating a dict with smoothened output, where output variable = maximum amplitude of voltage traces ###
    maxValue = findMaxDict( voltageTrace,marginOfBaseLine,marginOfInterest, smootheningWindow)
    ### Mapping this dict onto the random stimulation box here ####

    randStimBox = empty_2D_Array( randStimBoxSize)
    randStimBox = mapOutputToRandomStimulation(maxValue, coords, randStimBox, SizeOfPhotoactiveGrid)
    ################################################################
    randStimBoxNumStims, randStimBoxMaximum, randStimBoxStdDev = mappedMatrixStatistics(coords, randStimBox)

    ### Plotting heatmap of the random stimulation box here ### 
    title = 'Averaged maximum recorded voltage during random stimulation for each square'
    plotHeatMapBox(randStimBoxMaximum, title)

    title = 'Histogram of averaged maximum recorded voltage during random stimulation for each square'
    plotHistogram(randStimBoxMaximum,title)
    ### Plotting heatmap of the response variability here ### 
    title = 'Variability in response during random stimulation for each square'
    plotHeatMapBox(randStimBoxStdDev, title)

    ### Plotting heatmap of the frequency of photostimulation here ### 
    title = 'The times a given square on the grid is photostimulation squares'
    plotHeatMapBox(randStimBoxNumStims, title)

if __name__ == '__main__':
    main()
