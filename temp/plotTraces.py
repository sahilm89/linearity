'''General Analysis and plotting file for voltage traces from Aanchal's
Random Simulation project '''

from util import readMatlabFile, parseDictKeys, find_BaseLine_and_WindowOfInterest_Margins, convertDictToList,plotTrace, checkBaseLineStability, bandpass_ifft, besselFilter, fitDoubleExpToEPSP, doubleExponentialFunction,acorr 
import matplotlib.pyplot as plt
import sys
import copy
import numpy as np

file = sys.argv[1]
print file

#file = '/home/sahil/Documents/Codes/bgstimPlasticity/data/august/160830/c1/CPP/CPP.mat'
#file = '/home/sahil/Documents/Codes/bgstimPlasticity/photoDiode/160205_projector/Sahils_code_v3/10sec.mat'

samplingTime = 0.05 # ms

threshold = 0.05  # 50 mV 
smootheningTime = 0.25 # ms
baseline= 100. # ms
interest= 50. # ms
Low_cutoff, High_cutoff, F_sample = 0., 200., 20000.


smootheningWindow = int(smootheningTime/samplingTime )
baselineWindowWidth = int(baseline/samplingTime)
interestWindowWidth = int(interest/samplingTime)

dictMat = readMatlabFile(file)
volt, photodiode = parseDictKeys(dictMat)
#print volt, photodiode

# Photodiode trace code
#for key in volt.keys()[:2]
#    plt.plot(volt[key].T[0], volt[key].T[1])
#plt.show()


baseLineWindow, interestWindow = find_BaseLine_and_WindowOfInterest_Margins(photodiode, threshold, baselineWindowWidth, interestWindowWidth)

print baseLineWindow, interestWindow

first = 0
numTraces =3
trace_unfiltered  = convertDictToList(volt)

for key in volt.keys():
    Filtered_signal = besselFilter(volt[key].T[1], cutoff=High_cutoff, F_sample=F_sample)
    volt[key].T[1] = Filtered_signal

#baseLineAveraged = checkBaseLineStability(volt, baseLineWindow)
#plt.plot(baseLineAveraged)
#plt.show()

trace = convertDictToList(volt)
numTraces = len(trace)
#numTraces =2 
#print numTraces
print all(trace) == all(trace_unfiltered)
#outFile = '/home/sahil/Documents/Codes/bgstimPlasticity/data/august/160223/c2/CPP/5_rawTraces.png'
#plotTrace(trace, baseLineWindow, outFile)

plotTrace(trace_unfiltered[first:first + numTraces] + trace[first:first + numTraces], baseLineWindow, interestWindow)

#plotTrace(, baseLineWindow)

#trial_trace = np.array(trace[1][1][interestWindow[0]:interestWindow[1]])
#fig, ax = plt.subplots()
#acorr(trial_trace)
#ax.plot(trial_trace/trial_trace.max(), 'ro')
#plt.show()
