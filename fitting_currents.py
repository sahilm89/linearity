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

        model.set_param_hint('tOn', value =max(time)/5.1 , min = 0., max = max(time))
        model.set_param_hint('t_ratio', value =10., min=1.05)
        model.set_param_hint('tOff', min = 0., expr='tOn*t_ratio')
        model.set_param_hint('t_peak', expr = 't_0 + ((tOff * tOn)/(tOff-tOn)) * log(tOff/tOn)')
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

for type in neuron.experiment.keys():
    #if type == 2:
        for numSquares in neuron.experiment[type].keys(): 
            print "Processing {} of {}".format(numSquares, type)
            trials = neuron.experiment[type][numSquares].trial
            for i in trials:
                #avg_psp = np.mean(trials[i].interestWindow)
                #normalized_interestWindow = trials[i].interestWindow/avg_psp
                if type == 1:
                    normalized_interestWindow = -1e9*(trials[i].interestWindow) # Picoamperes and changing signs to positive
                else:
                    normalized_interestWindow = 1e9*(trials[i].interestWindow) # Picoamperes and changing signs to positive
                time = np.arange(len(trials[i].interestWindow))*trials[i].samplingTime
                if trials[i].experiment.type == 1 or trials[i].experiment.type == 2:
                    print trials[i].feature[6], 1e9*trials[i].feature[0]
                    result = fitFunctionToPSP(time, normalized_interestWindow, trials[i].experiment.type)
                    if result.redchi <= 0.1:
                        trials[i].fit = result.params
                        #trials[i].fit["g_max"].value*= avg_psp 
                    else:
                        trials[i].fit = None 
                print "{} done.".format(i)

        print "Processed all for {} of type {}".format(numSquares, type)
with open(plotDir + '/' + neuron.index + '_fits.pkl', 'w') as input:
    neuron = pickle.dump(neuron, input, pickle.HIGHEST_PROTOCOL)
