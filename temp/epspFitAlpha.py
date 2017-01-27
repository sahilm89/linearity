from util import readMatlabFile, parseDictKeys, find_BaseLine_and_WindowOfInterest_Margins, convertDictToList, alphaFunction, normalizeToBaseLine 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file = '/home/sahil/Documents/Codes/bgstimPlasticity/data/CPP.mat'

samplingTime = 0.05 # ms
threshold = 0.05  # 50 mV 
smootheningTime = 0.25 # ms
baseline= 100. # ms
interest= 50. # ms

smootheningWindow = int(smootheningTime/samplingTime )
baselineWindowWidth = int(baseline/samplingTime)
interestWindowWidth = int(interest/samplingTime)

dict = readMatlabFile(file)
volt, photodiode = parseDictKeys(dict)
trace = convertDictToList(volt)

baseLineWindow, interestWindow = find_BaseLine_and_WindowOfInterest_Margins(photodiode, threshold, baselineWindowWidth, interestWindowWidth)
############################################################################

normalized_Trace = [normalizeToBaseLine(traceIndex[1], baseLineWindow) for traceIndex in trace]

voltageTrace = [traceIndex[interestWindow[0]+180:interestWindow[1]+180] for traceIndex in normalized_Trace]

time = np.arange(0.,100., 100./1000.) 
popt, pcov = curve_fit(alphaFunction, time, voltageTrace[0])
#trace[0][0][interestWindow[0]:interestWindow[1]] 
plt.plot( time, voltageTrace[0])
plt.plot( time, alphaFunction(time, popt))

plt.show()
for trace in voltageTrace:
    popt, pcov = curve_fit(alphaFunction, time, trace)
    plt.plot(time, trace)
    plt.plot(time, alphaFunction(time,popt))
plt.show()
