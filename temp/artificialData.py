''' Create Artificial Data to replicate random stimulation'''
from util import *
from itertools import cycle

measureList = {1:'maximum', 2:'area under the curve'}
measure = 1 
frameRate = 5e-5 # frame rate is 20kHz
numRepeats = 5
marginOfBaseLine = [0,2000]
marginOfInterest = [2000,4000]

gridShape= (15,15)
#N = randomMatrix(gridShape) # Synaptic Weight matrix (pre-synaptic grid shape)
N = randomLogNormalMatrix(gridShape) # Synaptic Weight matrix (pre-synaptic grid shape) lognormally distributed
totalGridSize = len(N.flatten())

photostimSequence = []
gridElements = [ (x,y) for x in range(gridShape[0]) for y in range(gridShape[1]) ]

for repeat in range(numRepeats):
    np.random.shuffle(gridElements)
    photostimSequence.extend(gridElements)

def convertListToTimeDict(time,traces):
    signalDict = {}
    keyRange = range(len(traces))
    for trace,key in zip(traces,keyRange):
        sig = np.array([[t,s] for t,s in zip(time, trace)])
        signalDict.update({key+1 : sig})
    return signalDict

def projectionMatrix(photoStimSequence,numPhotoStimSquares,numTrials, gridSize):
    ''' It returns the matrix of projection vectors'''
    xindex = []
    yindex = []
    projectionMat = []

    for firstIndex in range(numTrials):
        try : (firstIndex + 1) * numPhotoStimSquares < len(photostimSequence)
        except IndexError:
            raise Exception("**ERROR** Number of trials ran longer than the predefined photostimulation coords.")

        projIndices = photoStimSequence[firstIndex*numPhotoStimSquares : (firstIndex + 1)* numPhotoStimSquares]

        projection = vectorOfZeros(gridSize[0]*gridSize[1])

        for index in projIndices:
            xindex.append(index[0])
            yindex.append(index[1])

            onedimIndex = index[0]*gridSize[1] + index[1]

            projection[onedimIndex] = 1
        projectionMat.append(projection)
    coords =  tuple([xindex,yindex])
    return projectionMat, coords

def main():

    smootheningWindow = 1
   
    projection_1sqr = [] # The photostimulation projection sequence

    numTrials = 800 
    SizeOfPhotoactiveGrid = 1
    randStimBox = empty_2D_Array( gridShape)
    synapseParameters = [0.4, 5.] # Double exponential
    #synapseParameters = [10.] # Single exponential alpha function
    projection_1sqr, coords = projectionMatrix(photostimSequence, SizeOfPhotoactiveGrid,numTrials, gridShape)

    signal_1sqr = createArtificialData(numTrials,N,projection_1sqr,synapseParameters,SizeOfPhotoactiveGrid)
    time_1sqr = np.arange(len(signal_1sqr[0]))*frameRate

    voltageTraceDict_1sqr = convertListToTimeDict(time_1sqr, signal_1sqr) 

    for i in range(len(signal_1sqr)):
        plt.plot(signal_1sqr[i])
    plt.show()
    
    mappingValue = findMaxDict( voltageTraceDict_1sqr,marginOfBaseLine,marginOfInterest, smootheningWindow)
    measure_1sq = mappingValue
    randStimBox = mapOutputToRandomStimulation(mappingValue, coords, randStimBox, SizeOfPhotoactiveGrid)

    randStimBoxNumStims_1, randStimBoxMaximum_1, randStimBoxStdDev_1 = mappedMatrixStatistics(coords, randStimBox)

###################################################################
    projection_5sqr = []
    SizeOfPhotoactiveGrid = 5
    numTrials = int(numTrials/SizeOfPhotoactiveGrid)
    #randStimBox = empty_2D_Array( gridShape)
    randStimBox = randStimBoxMaximum_1 

    projection_5sqr, coords = projectionMatrix(photostimSequence, SizeOfPhotoactiveGrid, numTrials, gridShape)

    signal_5sqr = createArtificialData(numTrials,N, projection_5sqr, synapseParameters, SizeOfPhotoactiveGrid)
    time_5sqr = np.arange(len(signal_5sqr[0]))*frameRate

    voltageTraceDict_5sqr = convertListToTimeDict(time_5sqr, signal_5sqr) 

    for i in range(len(signal_5sqr)):
        plt.plot(signal_5sqr[i])
    plt.show()
    
    mappingValue = findMaxDict( voltageTraceDict_5sqr,marginOfBaseLine,marginOfInterest, smootheningWindow)
    measure_5sq = mappingValue
    randStimBox = mapOutputToRandomStimulation(mappingValue, coords, randStimBox, SizeOfPhotoactiveGrid)

    randStimBoxNumStims_5, randStimBoxMaximum_5, randStimBoxStdDev_5 = mappedMatrixStatistics(coords, randStimBox)
 
    ## Plotting histograms to compare:
    plt.hist(randStimBoxMaximum_1.flatten(),bins=100,color='blue',label='1sq')
    plt.hist(randStimBoxMaximum_5.flatten(),bins=100,color='red',label='5sq')
    plt.legend()
    plt.show()

    ### Plotting heatmap of the random stimulation box here ### 
    title = 'Weighted contributions for ' + measureList[measure] +' of the recorded voltage of each square in 1 square stimulation'
    plotHeatMapBox(randStimBoxMaximum_1, title)

    title = 'Weighted contributions for ' + measureList[measure] +' of the recorded voltage of each square in 5 square stimulation'
    plotHeatMapBox(randStimBoxMaximum_5, title)
    
    ############ Plotting scatter plots for comparison ##############
    x,y = randStimBoxMaximum_1.flatten(),randStimBoxMaximum_5.flatten()
    axisWidth = (0.,1.05*max(max(x),max(y)))
    plt.scatter(x,y)
    plt.plot(axisWidth, axisWidth, 'r--' )
    plt.xlim(axisWidth)
    plt.ylim(axisWidth)
    plt.xlabel('Average of the measure for single sqr. stimulation')
    plt.ylabel('Weighted single sqr. contributions for 5 sqr. stimulation')
    plt.annotate('Supra-linear summation', xy=(0.70*axisWidth[1], 0.70*axisWidth[1] ), xytext=(0.60*axisWidth[1],0.80*axisWidth[1]), arrowprops=dict(arrowstyle='<-'), ha='center', va='center')
    plt.annotate('Sub-linear summation', xy=(0.70*axisWidth[1],0.70*axisWidth[1] ), xytext=(0.80*axisWidth[1], 0.60*axisWidth[1]), arrowprops=dict(arrowstyle='<-'), ha='center', va='center')
    plt.suptitle('Single square vs Contribution from 5 sqr weight-mapped to 1 sqr')
    plt.title(measureList[measure])
    plt.show()
    #plt.savefig('1_square_domain_scatterplot')
    plt.close()
    ############################################
    
    ############### Adding up measures here:###
    ########### doing the adding up measures here #############
    meanMat_5square = randStimBoxMaximum_5
    meanMat_1square = randStimBoxMaximum_1
    
    list_5sqr = convert_1dim_DictToList(measure_5sq)
    totalResponses = len(list_5sqr)
    addedResponse = addRandomStimulationResponses(coords, meanMat_1square, SizeOfPhotoactiveGrid, totalResponses)
    ############################################################
    
    ############ Plotting scatter plots for comparison ##############
    x,y = addedResponse,list_5sqr
    axisWidth = (0,1.05*max(max(x),max(y)))
    plt.scatter(x,y)
    plt.plot(axisWidth, axisWidth, 'r--' )
    plt.xlim(axisWidth)
    plt.ylim(axisWidth)
    plt.xlabel('Summed up response for 5 squares for single square stimulation')
    plt.ylabel('Maximum for five square stimulation')
    plt.annotate('Supra-linear summation', xy=(0.70*axisWidth[1], 0.70*axisWidth[1] ), xytext=(0.60*axisWidth[1],0.80*axisWidth[1]), arrowprops=dict(arrowstyle='<-'), ha='center', va='center')
    plt.annotate('Sub-linear summation', xy=(0.70*axisWidth[1],0.70*axisWidth[1] ), xytext=(0.80*axisWidth[1], 0.60*axisWidth[1]), arrowprops=dict(arrowstyle='<-'), ha='center', va='center')
    plt.suptitle('Summed up single square output versus 5 square output')
    plt.title(measureList[measure])
    plt.show()
    #plt.savefig('5_square_domain_scatterplot')
    plt.close()
    
    ############ Plotting histogram of measure ##############
    plt.hist(list_5sqr,bins=50)
    plt.title ("Distribution of " + measureList[measure] )
    plt.xlabel(measureList[measure] + ' of voltage trace in the window of interest')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == '__main__':
    main()
