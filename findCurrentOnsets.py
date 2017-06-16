#!/usr/bin/python2.7
from Linearity import Neuron
import sys
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
n = Neuron.load(sys.argv[1])

def findOnsetTime(trial, step=2., slide = 0.05, minOnset = 2., maxOnset = 50., initpValTolerance=0.5):
    maxIndex = int(trial.F_sample*maxOnset*1e-3)
    if expType == 1:
        maxOnsetIndex = np.argmax(-trial.interestWindow[:maxIndex])
    elif expType == 2:
        maxOnsetIndex = np.argmax(trial.interestWindow[:maxIndex])
    else:
        maxOnsetIndex = np.argmax(trial.interestWindow[:maxIndex])
    
    window_size = len(trial.interestWindow)
    step_size = int(trial.F_sample*step*1e-3)
    
    overlap =  int(trial.F_sample*0.05*1e-3)
    
    index_right = maxOnsetIndex
    index_left = index_right - step_size
    minOnsetIndex = int(trial.F_sample*minOnset*1e-3)
    
    baseMean = np.mean(trial.interestWindow[:minOnsetIndex])
    factor = 5 
    thresholdGradient = 0.01
    pValTolerance = initpValTolerance
    
    #if -baseMean*factor < trial.interestWindow[maxOnsetIndex] < baseMean*factor:
    #    return 0
    #print baseMean
    l_window = trial.interestWindow[:minOnsetIndex]
    while (index_left>minOnset):
        r_window = trial.interestWindow[index_left:index_right] #, trial.baselineWindow #trial.interestWindow[index_left - step_size:index_left]
    
        #if baseMean - 0.1 < np.mean(r_window) < baseMean + 0.1:
        #if (-factor*baseMean  < np.mean(r_window) < factor*baseMean) and (np.average(np.abs(np.gradient(r_window))) < thresholdGradient):
        
        stat, pVal = ss.ks_2samp(r_window, l_window)
        if pVal>pValTolerance:
            #print pVal, pValTolerance,float(index_right)/trial.F_sample
                        if (trial.experiment.type == 1):# and np.mean(trial.interestWindow[index_left:index_right]) >= baseMean) :
                            #return float(index_right + np.argmax(trial.interestWindow[index_right:maxOnsetIndex]))/trial.F_sample
                            smoothing = []
                            #for index in range(index_right, maxOnsetIndex-step_size+1):
                            #    smoothing.append(np.average(trial.interestWindow[index: index+step_size]))
                            if len(smoothing)>2:
                                return float(index_right + np.argmax(smoothing) + int(step_size/2))/trial.F_sample
                            else:
                                return float(index_right)/trial.F_sample
                            #return float(index_right + np.argmax(np.abs(np.gradient(trial.interestWindow[index_right:maxOnsetIndex]))))/trial.F_sample
                            #        return float(index) /trial.F_sample
                        elif (trial.experiment.type == 2):# and np.mean(trial.interestWindow[index_left:index_right]) <= baseMean):
                            #return float(index_right + np.argmin(trial.interestWindow[index_right:maxOnsetIndex]))/trial.F_sample
                            #return float(index_right + np.argmax(np.abs(np.gradient(trial.interestWindow[index_right:maxOnsetIndex]))))/trial.F_sample
                            smoothing = []
                            #for index in range(index_right, maxOnsetIndex-step_size+1):
                            #    smoothing.append(np.average(trial.interestWindow[index: index+step_size]))
                            if len(smoothing)>2:
                                return float(index_right + np.argmin(smoothing)+ int(step_size/2))/trial.F_sample
                                #return float(index_right + step_size*np.argmax(np.abs(np.gradient(smoothing))))/trial.F_sample
                            else:
                                return float(index_right)/trial.F_sample
        
                            #return float(index_right + step_size*np.argmax(np.abs(np.gradient(smoothing))))/trial.F_sample
                            #    if (np.average(trial.interestWindow[index: index+step_size]))> 5*baseMean:
                            #        return float(index) /trial.F_sample
                            #return float(index_right + np.argmax((trial.interestWindow[index_right:]>baseMean)) ) /trial.F_sample
                        else:
                            return float(index_right)/trial.F_sample
                        #return float(index_left+(step_size/2))/trial.F_sample
        else:
            index_left-=overlap
            index_right-=overlap
            if index_left<=minOnsetIndex:
                pValTolerance/=2
                #factor*=2
                #thresholdGradient*=2
                if pValTolerance<0.01:
                                print "{} pval too low for {} tolerance, increasing baseline size".format(pVal, pValTolerance)
                                minOnset*=2
                                #step_size*=2
                                index_right = maxOnsetIndex
                                index_left = maxOnsetIndex - step_size
        
                                l_window = trial.interestWindow[:minOnsetIndex]
                                pValTolerance = initpValTolerance 
        
                                if minOnsetIndex > maxOnsetIndex - step_size :
                                    print "Returning Nan"
                                    return np.nan
                else:
                    index_right = maxOnsetIndex
                    index_left = maxOnsetIndex - step_size

avg_exc_onset = {}
avg_inh_onset = {}
avg_exc_max = {}

for expType, exp in n:
    for sqr in exp:
        for coord in exp[sqr].coordwise:
            if expType == 1:
                avg_exc_onset[coord] = np.nanmean([findOnsetTime(trial) for trial in exp[sqr].coordwise[coord].trials])
                avg_exc_max[coord] = -exp[sqr].coordwise[coord].average_feature[5]
            if expType == 2:
                avg_inh_onset[coord] = np.nanmean([findOnsetTime(trial) for trial in exp[sqr].coordwise[coord].trials])
print (avg_exc_max, avg_exc_onset, avg_inh_onset)
delay, max = [], []
for coord in set(avg_exc_onset).intersection(set(avg_inh_onset)):
    delay.append(avg_inh_onset[coord]- avg_exc_onset[coord])
    max.append(avg_exc_max[coord])

fig, ax = plt.subplots()
ax.scatter(max, delay)
plt.show()
