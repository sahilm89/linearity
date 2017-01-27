'''
firstDict = {}
secondDict  = {}
for key in measure_1sq.keys():
    if key<226:
        firstDict.update({key:measure_1sq[key]})
    else:
        secondDict.update({key-225:measure_1sq[key]})

randStimBoxSize = (15,15)
randStimBox = empty_2D_Array(randStimBoxSize)

randStimBox_1 = mapOutputToRandomStimulation(firstDict, coords, randStimBox, SizeOfPhotoactiveGrid)
n,randStimBox_1,v = mappedMatrixStatistics(coords, randStimBox_1) 

randStimBox = empty_2D_Array(randStimBoxSize)
randStimBox_2 = mapOutputToRandomStimulation(secondDict, coords, randStimBox, SizeOfPhotoactiveGrid)
n,randStimBox_2,v = mappedMatrixStatistics(coords, randStimBox_2) 

title = 'heu'

plotHeatMapBox(randStimBox_1, title)

np.savetxt('oneFile.txt', randStimBox_1) 

plotHeatMapBox(randStimBox_2, title)
plotContourMap(randStimBox_2, title)
plotContourMap(meanMat_1square, title)
'''


