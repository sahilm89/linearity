import pickle
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

analysisFile = os.path.abspath(sys.argv[1])
plotPath = os.path.dirname(analysisFile)
with open(analysisFile, 'rb') as input:
    neuron = pickle.load(input)

exc = {}
control = {}
feature = 1
for type in neuron.experiment.keys():
    #for feature in neuron.features:
        for squares in neuron.experiment[type].keys(): 
            for index in neuron.experiment[type][squares].coordwise:
                if feature in neuron.experiment[type][squares].coordwise[index].feature:
                    print len(index), index
                    if type == "Control":
                        control.update({index :neuron.experiment[type][squares].coordwise[index].average_feature[feature]})
                    elif type=="GABAzine":
                        exc.update({index :neuron.experiment[type][squares].coordwise[index].average_feature[feature]})
                    else:
                        print "WTF?"
                else:
                    break
#print exc.keys(), control.keys()

exc_list, control_list = [], []
for key in exc.keys():
    if key in control.keys():
        exc_list.append( exc[key])
        control_list.append( control[key])

f, ax = plt.subplots(2,1)
ax[0].scatter(np.log10(exc_list), control_list)
ax[1].scatter(exc_list, control_list, c='g')
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
        if not numSquares == 1:
            c =next(color)
            trials = []
            expected, observed = [], []
            for coord in neuron.experiment[type][numSquares].coordwise.keys():
                print coord, len(neuron.experiment[type][numSquares].coordwise[coord].trials)
                for trial in neuron.experiment[type][numSquares].coordwise[coord].trials:
                    trials.append(trial.index)
                    if feature in trial.feature:
                        observed.append(trial.feature[feature])
                        expected.append(neuron.experiment[type][numSquares].coordwise[coord].expected_feature[feature])
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
plt.title(neuron.features[feature])
plt.savefig("{}/{}_scatter_averaged_both".format(plotPath, type)) 
#plt.show()
plt.close()
