import pickle
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.stats as ss

analysisFile = os.path.abspath(sys.argv[1])
plotPath = os.path.dirname(analysisFile)
with open(analysisFile, 'rb') as input:
    neuron = pickle.load(input)

exc = {}
inh = {}
feature = 0
for type in neuron.experiment.keys():
    #for feature in neuron.features:
        for squares in neuron.experiment[type].keys(): 
            for index in neuron.experiment[type][squares].coordwise:
                if feature in neuron.experiment[type][squares].coordwise[index].feature:
                    print index, [trial.index for trial in neuron.experiment[type][squares].coordwise[index].trials]
                    if type == 0:
                        inh.update({index :neuron.experiment[type][squares].coordwise[index].average_feature[feature]})
                    elif type == 1:
                        exc.update({index :neuron.experiment[type][squares].coordwise[index].average_feature[feature]})
                    else:
                        print "WTF?"
                else:
                    break

exc, exc_v = {}, {}
inh, inh_v = {}, {}
feature = 6
for type in neuron.experiment.keys():
    	new_index=0
    #for feature in neuron.features:
        for squares in neuron.experiment[type].keys(): 
            for index in neuron.experiment[type][squares].trial:
                if feature in neuron.experiment[type][squares].trial[index].feature:
                    #print index, [trial.index for trial in neuron.experiment[type][squares].coordwise[index].trials]
                    if type == 2:
                        inh.update({new_index :neuron.experiment[type][squares].trial[index].feature[feature]})
                        inh_v.update({new_index :neuron.experiment[type][squares].trial[index].interestWindow})
			new_index+=1
                    elif type == 1:
                        exc.update({new_index :neuron.experiment[type][squares].trial[index].feature[feature]})
                        exc_v.update({new_index :neuron.experiment[type][squares].trial[index].interestWindow})
			new_index+=1
                    else:
                        print "WTF?"
                else:
                    break

print exc.keys(), inh.keys()
print [exc[key] for key in exc.keys()]

#exc_volt = [exc_v[key] for key in exc_v.keys()]
exc_volt, exc_onset = zip(*[(exc_v[key], exc[key]) for key in exc.keys()])
inh_volt, inh_onset = zip(*[(inh_v[key], inh[key]) for key in inh.keys()])

#inh_volt = [inh_v[key] for key in inh_v.keys()]
#inh_onset = np.array([inh[key] for key in inh.keys()])

print exc_onset, inh_onset
print len(exc_volt), len(inh_volt)
F_sample = neuron.experiment[1][1].F_sample
numSkip = 1
color = iter(cm.viridis(np.linspace(0,1,len(exc_volt))[::numSkip]))

fig, ax = plt.subplots(len(exc_volt[::numSkip]), 1)
with PdfPages(plotPath + '/' + 'EI_all_onsets.pdf') as pdf:
    for iter, (e,i) in enumerate(zip(exc_volt[::numSkip],inh_volt[::numSkip])):
        c=next(color)
        fig,ax = plt.subplots()
        if (exc_onset[iter] and e[iter]): 
            time = np.arange(len(e))*1e3/F_sample
            ax.plot( time ,e*1e9, c='c', label="E")
            ax.plot([exc_onset[iter]*1e3], [e[int(exc_onset[iter]*20)]*1e9], marker='o', markersize=5, color='r', mfc='none')
            #ax.axvline(exc_onset[iter]*1e3, c='b')#c)
        if inh_onset[iter] and np.isnan(any(i[iter])):
            ax.plot( np.arange(len(i))*1e3/F_sample, i*1e9, c='m', label="I")
            ax.plot([inh_onset[iter]*1e3], [i[int(inh_onset[iter]*20)]*1e9], marker='o', markersize=5, color='b', mfc='none')
            #ax.axvline(inh_onset[iter]*1e3, c='g')#)
        plt.xlabel("Time(ms)")
        plt.ylabel("Current(pA)")
#ax.scatter(exc_onset,inh_onset )
        pdf.savefig()
        plt.close(fig)

fig, ax = plt.subplots()
ax.scatter(exc_onset, inh_onset)
plt.show()

exc_volt, exc_onset, inh_volt, inh_onset = zip(*[(exc_v[key], exc[key], inh_v[key], inh[key]) for key in exc.keys()])

delay, volt = [], []
assert len(exc_volt) == len(exc_onset)
for i,e,v in zip(inh_onset, exc_onset, exc_volt):
    if e and i:
        delay.append(i-e)
        volt.append(min(v)*-1e9)
print len(volt), len(delay)
fig, ax = plt.subplots()
#vspace = np.linspace(0, max(volt), 1000)
#line1 = np.poly1d(np.polyfit(volt, delay, 1))
#line2 = np.poly1d(np.polyfit(volt, delay, 2))
#ax.plot(vspace, line1(vspace), label= "1st order")
#ax.plot(vspace, line2(vspace), label= "2nd order")
#print vspace, np.polyfit(volt, delay, 1), line2(vspace)
ax.scatter(volt, delay)
ax.set_title("Delay decreases with increase in max exc current")
ax.set_xlabel("Max Current (pA)")
ax.set_ylabel("Delay between E and I")
plt.legend()
plt.savefig(plotPath + '/' + 'Delay_max.png')
plt.close()


exc_list, inh_list = [], []
for key in exc.keys():
    if key in inh.keys():
        exc_list.append( exc[key])
        inh_list.append( inh[key])
f, ax = plt.subplots(2,1)
ax[0].scatter(np.log10(exc_list), inh_list)
ax[1].scatter(exc_list, inh_list, c='g')
plt.show()

color=iter(plt.cm.viridis(np.linspace(0,1,len(neuron.experiment[type]))))


for type in neuron.experiment.keys():
    #for squares in neuron.experiment[type].keys(): 
    #    trial_features = []
    #    for index in neuron.experiment[type][squares].trial:
    #        if feature in neuron.experiment[type][squares].trial[index].feature: # Checking if features are missing due to flags.
    #            trial_features.append(neuron.experiment[type][squares].trial[index].feature[feature])
    #    plt.scatter(range(len(trial_features)), trial_features, label=str(squares), c=next(color)) 
    #plt.title(neuron.features[feature])
    #plt.xlabel("Trial Number")
    #plt.ylabel("Amplitude")
    #plt.legend(loc='upper right', bbox_to_anchor=(1,1))
    ##plt.tight_layout()
    #plt.show()       

############################## Trial Checking ###############

    color=iter(plt.cm.viridis(np.linspace(0,1,len(neuron.experiment[type]))))
    ax = plt.subplot(111)
    for numSquares in neuron.experiment[type].keys(): 
    #for numSquares in [2]: 
        if not numSquares == 1:
            c =next(color)
            expected, observed = [], []
            for coord in neuron.experiment[type][numSquares].coordwise.keys():
                if feature in neuron.experiment[type][numSquares].coordwise[coord].feature:
                    observed.append(neuron.experiment[type][numSquares].coordwise[coord].average_feature[feature])
                    expected.append(neuron.experiment[type][numSquares].coordwise[coord].expected_feature[feature])
            E = np.array(expected)
            O = np.array(observed)
            ax.scatter(E,O, c = c)
            slope, intercept  = neuron.experiment[type][numSquares].regression_coefficients[feature]['slope'], neuron.experiment[type][numSquares].regression_coefficients[feature]['intercept']
            ynew = slope*E + intercept
            ax.plot(E, ynew, c=c, label='{},m= {:.2f}'.format(numSquares, slope))
    axisWidth = (min(min(E), min(O)), max(max(E), max(O)))
    ax.plot(axisWidth, axisWidth, 'r--')
    plt.legend()
    plt.xlabel('Expected (mV)')
    plt.ylabel('Observed (mV)')
    plt.title(neuron.features[feature])
    plt.savefig("{}/{}_scatter_averaged".format(plotPath, type)) 
    #plt.show()
    plt.close()

    color=iter(plt.cm.viridis(np.linspace(0,1,len(neuron.experiment[type]))))
    ax = plt.subplot(111)
    for numSquares in neuron.experiment[type].keys(): 
        if not numSquares == 1 and not numSquares > 2:
            c =next(color)
            trials = []
            expected, observed = [], []
            for coord in neuron.experiment[type][numSquares].coordwise.keys():
                print "Coord: {}, expected: {}".format(coord, neuron.experiment[type][numSquares].coordwise[coord].expected_feature[feature])
                for trial in neuron.experiment[type][numSquares].coordwise[coord].trials:
                    trials.append(trial.index)
                    if feature in trial.feature:
                        observed.append(trial.feature[feature])
                        expected.append(trial.experiment.coordwise[coord].expected_feature[feature])
            E = np.array(expected)
            O = np.array(observed)
            ax.scatter(E,O, c = c)
            slope, intercept  = neuron.experiment[type][numSquares].regression_coefficients[feature]['slope'], neuron.experiment[type][numSquares].regression_coefficients[feature]['intercept']
            ynew = slope*E + intercept
            ax.plot(E, ynew, c=c, label='{},m= {:.2f}'.format(numSquares, slope))
            print type, len(trials), len(set(trials))

    plt.legend()
    plt.xlabel('Expected (mV)')
    plt.ylabel('Observed (mV)')
    axisWidth = (min(min(E), min(O)), max(max(E), max(O)))
    ax.plot(axisWidth, axisWidth, 'r--')

    plt.title(str(type) + " " + neuron.features[feature])
    #plt.show()
    plt.savefig("{}/{}_scatter_raw".format(plotPath, type)) 
    plt.close()

for type in neuron.experiment.keys():
    color=iter(plt.cm.viridis(np.linspace(0,1,len(neuron.experiment[type]))))
    ax = plt.subplot(111)
    for numSquares in neuron.experiment[type].keys(): 
        if not numSquares == 1:
            c =next(color)
            expected, observed = [], []
            for coord in neuron.experiment[type][numSquares].coordwise.keys():
                if feature in neuron.experiment[type][numSquares].coordwise[coord].feature:
                    observed.append(neuron.experiment[type][numSquares].coordwise[coord].average_feature[feature])
                    expected.append(neuron.experiment[type][numSquares].coordwise[coord].expected_feature[feature])
            E = np.array(expected)
            O = np.array(observed)
            ax.scatter(E,O, c = c)
            slope, intercept  = neuron.experiment[type][numSquares].regression_coefficients[feature]['slope'], neuron.experiment[type][numSquares].regression_coefficients[feature]['intercept']
            ynew = slope*E + intercept
            ax.plot(E, ynew, c=c, label='{},m= {:.2f}'.format(numSquares, slope))
plt.legend()
plt.xlabel('Expected (mV)')
plt.ylabel('Observed (mV)')
axisWidth = (min(min(E), min(O)), max(max(E), max(O)))
ax.plot(axisWidth, axisWidth, 'r--')
plt.title(neuron.features[feature])
plt.savefig("{}/{}_scatter_averaged_both".format(plotPath, type)) 
#plt.show()
plt.close()
