#!/usr/bin/python
''' This script just picks up the slope from the slope fitting files and creates a histogram of this data. This is to see if there is bimodality of slopes or not. '''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import numpy as np

oneSquare = '/home/sahil/analysis_linearity/*/areaUnderTheCurve_1sqr_slope.txt'
#oneSquare = '/home/sahil/*areaUnderTheCurve_1sqr_slope.txt'
fiveSquare = '/home/sahil/analysis_linearity/*/areaUnderTheCurve_5sqr_slope.txt'

#data/august/linearity/*/CPP/areaUnderTheCurve_5sqr_slope.txt'


one = glob.glob(oneSquare)
five = glob.glob(fiveSquare)

print one, five

oneArr = []
fiveArr = []

for oneFile,fiveFile in zip(one,five):
    oneArr.append(np.loadtxt(oneFile)[0])
    fiveArr.append(np.loadtxt(fiveFile)[0])

print oneArr
print fiveArr

plt.hist(oneArr)
plt.title("One Square")
plt.xlabel('Slope')
plt.ylabel('Frequency')
plt.savefig('histogram_1.png')
#plt.show()
plt.close()

plt.hist(fiveArr)
plt.title("Five Squares domain")
plt.xlabel('Slope')
plt.ylabel('Frequency')
plt.savefig('histogram_5.png')
#plt.show()
plt.close()
