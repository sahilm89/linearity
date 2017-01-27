#!/usr/bin/python
# This script creates the photostimulation circle for randomly stimulating CA3.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def makePhotoStimulationGrid(numEdgeSquares, skipSquaresBy=1):
    photoStimGrid = []
    for i in xrange(0,numEdgeSquares,skipSquaresBy):
        for j in xrange(0,numEdgeSquares,skipSquaresBy):
            photoStimGrid.append( (i,j))
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

numEdgeSquares = 13
circleDiameter = 11
skipSquaresBy = 2
diagonalOnly = True
circularGrid = makeCircularPhotostimulationGrid(numEdgeSquares,circleDiameter,skipSquaresBy=skipSquaresBy, diagonalOnly=diagonalOnly)

#numSquareRepeats = 2
#totalNumberOfSquares = numSquareRepeats * len(circularGrid)

totalNumberOfSquares = 12

print len(circularGrid), totalNumberOfSquares
circularRandomStimulationGrid = createRandomPhotoStimulation(totalNumberOfSquares, circularGrid)

print len(circularRandomStimulationGrid)

x,y = returnCoordsFromGrid(circularRandomStimulationGrid) 

with open("axonStimulation/randX.txt",'w') as coordFile:
    coordFile.write(','.join( [str(i) for i in x] ))

with open("axonStimulation/randY.txt",'w') as coordFile:
    coordFile.write(','.join( [str(i) for i in y] ))

with open("axonStimulation/CPP_randX.txt",'w') as coordFile:
    coordFile.write(','.join( [str(i) for i in x[:len(circularGrid)]] ))

with open("axonStimulation/CPP_randY.txt",'w') as coordFile:
    coordFile.write(','.join( [str(i) for i in y[:len(circularGrid)]] ))

#plt.hist(x, bins=max(x),alpha=0.5)

jointX = []
jointY = []
for k in range(len(x)):
    jointX.append(x[k*len(circularGrid):(k+1)*len(circularGrid)])

for k in range(len(y)):
    jointY.append(y[k*len(circularGrid):(k+1)*len(circularGrid)])

plt.hist(jointX,bins=max(x),alpha=0.5,stacked=True)
plt.hist(x,bins=max(x),alpha=0.5)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,range(len(x)),c=range(len(x)))
ax.set_xlim(0,numEdgeSquares)
ax.set_ylim(0,numEdgeSquares)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x,y,c=range(len(x)))
ax.set_xlim(0,numEdgeSquares)
ax.set_ylim(0,numEdgeSquares)
plt.show()
