import pickle
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

analysisFile = os.path.abspath(sys.argv[1])
with open(analysisFile, 'rb') as input:
    neuron = pickle.load(input)

exc = {}
control = {}

for type in neuron.experiment.keys():
    for feature in neuron.features:
        for squares in neuron.experiment[type].keys(): 
            for index in neuron.experiment[type][squares].coordwise:
                if feature in neuron.experiment[type][squares].coordwise[index].feature and feature == 0:
                    if type == "Control":
                        control.update({index :neuron.experiment[type][squares].coordwise[index].average_feature[feature]})
                    elif type=="GABAzine":
                        exc.update({index :neuron.experiment[type][squares].coordwise[index].average_feature[feature]})
                    else:
                        print "WTF?"
                else:
                    break
print exc, control

exc_list, control_list = [], []
for key in exc.keys():
    if key in control.keys():
        exc_list.append( exc[key])
        control_list.append( control[key])
plt.scatter(np.log10(exc_list), control_list)
plt.show()

color=iter(plt.cm.viridis(np.linspace(0,1,len(neuron.experiment[type]))))

for squares in neuron.experiment[type].keys(): 
    trial_features = []
    for index in neuron.experiment[type][squares].trial:
        if feature in neuron.experiment[type][squares].trial[index].feature: # Checking if features are missing due to flags.
            trial_features.append(neuron.experiment[type][squares].trial[index].feature[feature])
    plt.scatter(range(len(trial_features)), trial_features, label=str(squares), c=next(color)) 
plt.title(neuron.features[feature])
plt.xlabel("Trial Number")
plt.ylabel("Amplitude")
plt.legend(loc='upper right', bbox_to_anchor=(1,1))
#plt.tight_layout()
plt.show()       

############################## Trial Checking ###############

color=iter(plt.cm.viridis(np.linspace(0,1,len(neuron.experiment[type]))))
ax = plt.subplot(111)
feature = 0
for numSquares in neuron.experiment[type].keys(): 
#for numSquares in [2]: 
    if not numSquares == 1:
        c =next(color)
        expected, observed = [], []
        for coord in neuron.experiment[type][numSquares].coordwise.keys():
            if feature in neuron.experiment[type][numSquares].coordwise[coord].feature:
                observed.append(neuron.experiment[type][numSquares].coordwise[coord].average_feature[feature])
                expected.append(neuron.experiment[type][numSquares].coordwise[coord].expected_feature[feature])
        ax.scatter(expected,observed, c = c)
        slope, intercept  = neuron.experiment[type][numSquares].regression_coefficients[feature]['slope'], neuron.experiment[type][numSquares].regression_coefficients[feature]['intercept']
        ynew = slope*np.array(expected) + intercept
        ax.plot(expected, ynew, c=c, label='{},m= {:.2f}'.format(numSquares, slope))
plt.legend()
plt.xlabel('Expected')
plt.ylabel('Observed')
plt.title(neuron.features[feature])
plt.show()
