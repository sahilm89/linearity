#!/usr/env/python
import glob
import os
import numpy as np
from util import plotHeatMapBox
from util import normalize
from util import normalizeToBaseLine

INPUTDIR = '/data/aanchal/Patch- backed nov2015/patch_data/RS_Analysis/Patch Recording/*/*/CPP/*_1sqr.txt'
normalizeFlag = 1
showPlots = 0
outputDir = '/pantua/sahil/sahil_aanchal/bgstimPlasticity/averagePicture_allresponse/'

k = glob.glob(INPUTDIR)
matrices = []

for CPP in k:
    matrices.append (np.loadtxt(CPP) )

if normalizeFlag:
     newMatrices = [normalize(matrix) for matrix in matrices]
averageMatrix = np.mean(newMatrices, axis = 0) 
title = 'Average heatmap of all recordings'
plotHeatMapBox(averageMatrix, title, showPlots = showPlots, filename = outputDir + 'Average_heatmap' + '.png') 

i = 0
title = 'Heatmap of one recordings'
for newMatrix in newMatrices:
    plotHeatMapBox(newMatrix, title, showPlots = showPlots, filename = outputDir + 'Single_heatmap' + str(i) + '.png')
    i+=1
