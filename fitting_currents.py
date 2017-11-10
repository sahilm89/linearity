import pickle
import os
import sys
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

analysisFile = os.path.abspath(sys.argv[1])
plotDir = os.path.dirname(analysisFile)
with open(analysisFile, 'rb') as input:
    neuron = pickle.load(input)

def fitFunctionToPSP(time, vector, type, t_0=0, g_max=0):
    ''' Fits using lmfit '''

    def _doubleExponentialFunction(t, t_0, tOn, tOff, g_max):
        ''' Returns the shape of an EPSP as a double exponential function '''
        tPeak = t_0 + float(((tOff * tOn)/(tOff-tOn)) * np.log(tOff/tOn))
        A = 1./(np.exp(-(tPeak-t_0)/tOff) - np.exp(-(tPeak-t_0)/tOn))
        g = [ g_max * A * (np.exp(-(t_point-t_0)/tOff) - np.exp(-(t_point-t_0)/tOn)) if  t_point >= t_0 else 0. for t_point in t]
        return np.array(g)

    def _doubleExponentialFunction2(t, t_0_2, tOn_2, tOff_2, g_max_2):
        ''' Returns the shape of an EPSP as a double exponential function '''
        tPeak = t_0_2 + float(((tOff_2 * tOn_2)/(tOff_2-tOn_2)) * np.log(tOff_2/tOn_2))
        A = 1./(np.exp(-(tPeak-t_0_2)/tOff_2) - np.exp(-(tPeak-t_0_2)/tOn_2))
        g = [ g_max_2 * A * (np.exp(-(t_point-t_0_2)/tOff_2) - np.exp(-(t_point-t_0_2)/tOn_2)) if  t_point >= t_0_2 else 0. for t_point in t]
        return np.array(g)

    if type == 1 or type == 2:
        function = _doubleExponentialFunction 
        model = lmfit.Model(function)
        # Fixing values of variables from data
        # Onset time
        if not t_0:
            model.set_param_hint('t_0', value =max(time)/10., min=0., max = max(time))
        else:
            model.set_param_hint('t_0', value = t_0, vary=False)
        # g_max 
        if not g_max:
            model.set_param_hint('g_max', value = max(vector)/10., min = 0., max = max(vector))
        else:
            model.set_param_hint('g_max', value = g_max, vary=False)

        model.set_param_hint('tOn', value = max(time)/10. , min = 0., max = max(time))
        #model.set_param_hint('t_ratio', value =10., min=1.05)
        model.set_param_hint('tOff', value = max(time)/5., min = 0., max = max(time))
        #model.set_param_hint('t_peak', expr = 't_0 + ((tOff * tOn)/(tOff-tOn)) * log(tOff/tOn)')
        pars = model.make_params()

    result = model.fit(vector, pars, t=time)
    #print result.fit_report()
    #ax = plt.subplot(111)
    #ax.plot(time, vector, alpha=0.2)
    #ax.set_xlabel("Time")
    #ax.set_ylabel("mean normalized V")
    #ax.set_title("Double Exponential fit")
    #ax.plot(time, result.best_fit, '-')
    #plt.show()
    return result

for expType in neuron.experiment.keys():
    if expType in [1,2]:
        for numSquares in neuron.experiment[expType].keys(): 
            print "Processing {} of {}".format(numSquares, expType)
            trials = neuron.experiment[expType][numSquares].trial
            for i in trials:
                #avg_psp = np.mean(trials[i].interestWindow)
                #normalized_interestWindow = trials[i].interestWindow/avg_psp
                if expType == 1:
                    normalizingFactor = np.max(-trials[i].interestWindow)
                    normalized_interestWindow = -trials[i].interestWindow/normalizingFactor # Picoamperes and changing signs to positive
                if expType == 2:
                    normalizingFactor = np.max(trials[i].interestWindow)
                    normalized_interestWindow = trials[i].interestWindow/normalizingFactor # Picoamperes and changing signs to positive
                time = np.arange(len(trials[i].interestWindow))*trials[i].samplingTime

                result = fitFunctionToPSP(time, normalized_interestWindow, trials[i].experiment.type)
                if result.bic <= -10000:
                    trials[i].fit = result.params
                    trials[i].fit["g_max"].value*= normalizingFactor 
                else:
                    trials[i].fit = None 

                print "{} done.".format(i)

        print "Processed all for {} of type {}".format(numSquares, expType)
with open(plotDir + '/' + neuron.index + '_fits.pkl', 'w') as input:
    neuron = pickle.dump(neuron, input, pickle.HIGHEST_PROTOCOL)
