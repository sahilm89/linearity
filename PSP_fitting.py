import pickle
import os
import sys
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

analysisFile = os.path.abspath(sys.argv[1])
with open(analysisFile, 'rb') as input:
    neuron = pickle.load(input)

def fitFunctionToPSP(time, vector, type):
    ''' Fits using lmfit '''

    def _doubleExponentialFunction(t, t_0, tOn, tOff, g_max):
        ''' Returns the shape of an EPSP as a double exponential function '''
        tPeak = t_0 + float(((tOff * tOn)/(tOff-tOn)) * np.log(tOff/tOn))
        A = 1./(np.exp(-(tPeak-t_0)/tOff) - np.exp(-(tPeak-t_0)/tOn))
        g = [ g_max * A * (np.exp(-(t_point-t_0)/tOff) - np.exp(-(t_point-t_0)/tOn)) if  t_point >= t_0 else 0.  for t_point in t]
        return np.array(g)

    def _doubleExponentialFunction2(t, t_0_2, tOn_2, tOff_2, g_max_2):
        ''' Returns the shape of an EPSP as a double exponential function '''
        tPeak = t_0_2 + float(((tOff_2 * tOn_2)/(tOff_2-tOn_2)) * np.log(tOff_2/tOn_2))
        A = 1./(np.exp(-(tPeak-t_0_2)/tOff_2) - np.exp(-(tPeak-t_0_2)/tOn_2))
        g = [ g_max_2 * A * (np.exp(-(t_point-t_0_2)/tOff_2) - np.exp(-(t_point-t_0_2)/tOn_2)) if  t_point >= t_0_2 else 0.  for t_point in t]
        return np.array(g)

    if type=="GABAzine":
        function = _doubleExponentialFunction 
        model = lmfit.Model(function)
        model.set_param_hint('t_0', value =max(time)/10., min=0., max = max(time))
        model.set_param_hint('tOn', value =max(time)/5.1 , min = 0., max = max(time))
        model.set_param_hint('t_ratio', value =10., min=1.05)
        model.set_param_hint('tOff', min = 0., expr='tOn*t_ratio')
        model.set_param_hint('g_max', value = max(vector)/10., min = 0., max = max(vector))
        model.set_param_hint('t_peak', expr = 't_0 + ((tOff * tOn)/(tOff-tOn)) * log(tOff/tOn)')
        pars = model.make_params()

    elif type=="Control":
        function1 = _doubleExponentialFunction 
        function2 = _doubleExponentialFunction2 
        model = lmfit.Model(function1, pre='exc') - lmfit.Model(function2, pre='inh')
        model.set_param_hint('t_0', value =max(time)/10., min=0., max = max(time))
        model.set_param_hint('tOn', value =max(time)/5.1 , min = 0., max = max(time))
        model.set_param_hint('t_ratio', value =10., min=1.05)
        model.set_param_hint('tOff', min = 0., expr='tOn*t_ratio')
        model.set_param_hint('g_max', value = max(vector)/10., min = 0.)
        model.set_param_hint('t_peak', expr = 't_0 + ((tOff * tOn)/(tOff-tOn)) * log(tOff/tOn)')
 
        model.set_param_hint('delta', value =max(time)/8., min=0., max = max(time))
        model.set_param_hint('t_0_2', min = 0., max = max(time), expr='t_0 + delta')
        model.set_param_hint('tOn_2', value =max(time)/5.2 , min = 0., max = max(time))
        model.set_param_hint('t_ratio_2', value =10., min=1.05)
        model.set_param_hint('tOff_2', min = 0., expr='tOn_2*t_ratio_2')
        model.set_param_hint('g_max_2', value = max(vector)/6., min = 0.)
        model.set_param_hint('t_peak_2', expr = 't_0_2 + ((tOff_2 * tOn_2)/(tOff_2-tOn_2)) * log(tOff_2/tOn_2)')
        pars = model.make_params()

    result = model.fit(vector, pars, t=time)
    
    #ax = plt.subplot(111)
    #ax.plot(time, vector, alpha=0.2)
    #ax.set_xlabel("Time")
    #ax.set_ylabel("mean normalized V")
    #ax.set_title("Double Exponential fit")
    #ax.plot(time, result.best_fit, '-')
    #plt.show()
    return result

for type in neuron.experiment.keys():
    for numSquares in neuron.experiment[type].keys(): 
        print "Processing {} of {}".format(numSquares, type)
        trials = neuron.experiment[type][numSquares].trial
        for i in trials:
            avg_psp = np.mean(trials[i].interestWindow)
            normalized_interestWindow = trials[i].interestWindow/avg_psp
            time = np.arange(len(trials[i].interestWindow))*trials[i].samplingTime
            if trials[i].experiment.type == "Control":
                continue
                result = fitFunctionToPSP(time, normalized_interestWindow, trials[i].experiment.type)
                if result.redchi <= 0.1:
                    trials[i].fit = result.params
                    trials[i].fit["g_max"].value*= avg_psp 
                    trials[i].fit["g_max_2"].value*= avg_psp 
                else:
                    trials[i].fit = None 
            elif trials[i].experiment.type == "GABAzine":
                result = fitFunctionToPSP(time, normalized_interestWindow, trials[i].experiment.type)
                if result.redchi <= 0.1:
                    trials[i].fit = result.params
                    trials[i].fit["g_max"].value*= avg_psp 
                else:
                    trials[i].fit = None 
            print "{} done.".format(i)

print "Processed all".format(numSquares, type)
with open('/home/sahil/Documents/Codes/bgstimPlasticity/data/august/161013/c1/plots/c1_try.pkl', 'w') as input:
    neuron = pickle.dump(neuron, input, pickle.HIGHEST_PROTOCOL)
