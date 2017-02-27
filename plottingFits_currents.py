import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys

analysisFile = os.path.abspath(sys.argv[1])
plotDir = os.path.dirname(analysisFile)

with open (analysisFile,'rb') as p:
    neuron = pickle.load(p)

for type in neuron.experiment:
    if type == 1:
        e_t_0, e_t_on, e_t_off, e_g_max, e_t_peak, e_sq  = [], [], [], [], [], []
        g_max, t_o_e=  [], []
        for numSquares in neuron.experiment[type].keys():	
            for trialNum in neuron.experiment[type][numSquares].trial:
                print neuron.experiment[type][numSquares].trial[trialNum].fit
                if not neuron.experiment[type][numSquares].trial[trialNum].fit == None:
                    e_t_0.append(neuron.experiment[type][numSquares].trial[trialNum].fit["t_0"].value)
                    e_g_max.append(neuron.experiment[type][numSquares].trial[trialNum].fit["g_max"].value)
                    print neuron.experiment[type][numSquares].trial[trialNum].fit["g_max"].value
                    e_t_on.append(neuron.experiment[type][numSquares].trial[trialNum].fit["tOn"].value)
                    e_t_off.append(neuron.experiment[type][numSquares].trial[trialNum].fit["tOff"].value)
                    e_t_peak.append(neuron.experiment[type][numSquares].trial[trialNum].fit["t_peak"].value)
                    e_sq.append(numSquares)
            for coord in neuron.experiment[type][numSquares].coordwise:
                t_o_e.append(neuron.experiment[type][numSquares].coordwise[coord].average_feature[6])
                g_max.append(neuron.experiment[type][numSquares].coordwise[coord].average_feature[5])


    if type == 2:
        i_t_0, i_t_on, i_t_off, i_g_max, i_t_peak, i_sq  = [], [], [], [], [], []
        for numSquares in neuron.experiment[type].keys():	
            for trialNum in neuron.experiment[type][numSquares].trial:
                print neuron.experiment[type][numSquares].trial[trialNum].fit
                if not neuron.experiment[type][numSquares].trial[trialNum].fit == None:
                    i_t_0.append(neuron.experiment[type][numSquares].trial[trialNum].fit["t_0"].value)
                    i_g_max.append(neuron.experiment[type][numSquares].trial[trialNum].fit["g_max"].value)
                    i_t_on.append(neuron.experiment[type][numSquares].trial[trialNum].fit["tOn"].value)
                    i_t_off.append(neuron.experiment[type][numSquares].trial[trialNum].fit["tOff"].value)
                    i_t_peak.append(neuron.experiment[type][numSquares].trial[trialNum].fit["t_peak"].value)
                    i_sq.append(numSquares)
            #for coord in neuron.experiment[type][numSquares].coordwise:
                #i_t_o.append(neuron.experiment[type][numSquares].coordwise[coord].average_feature[6])



print e_g_max, i_g_max

plt.scatter(i_g_max, e_g_max)
plt.xlabel("$i_{max}$", size=20)
plt.ylabel("$e_{max}$", size=20)
plt.legend()
plt.show()

plt.scatter(np.array(i_t_0) - np.array(e_t_0), e_g_max)
plt.xlabel("$\delta$", size=20)
plt.ylabel("$e_{max}$", size=20)
plt.legend()
plt.show()

plt.scatter(e_t_off, e_t_0, c=i_sq, edgecolor='none')
plt.xlabel("$t_{off}$", size=20)
plt.ylabel("$t_{0}$", size=20)
plt.legend()
plt.show()

plt.scatter(i_g_max, i_t_0, c=i_sq, edgecolor='none')
plt.xlabel("$g_{max}$", size=20)
plt.ylabel("$t_{0}$", size=20)
plt.legend()
plt.show()

plt.scatter(e_t_off, e_t_on, c=i_sq, edgecolor='none')
plt.xlabel("$t_{off}$", size=20)
plt.ylabel("$t_{on}$", size=20)
plt.legend()
plt.show()

plt.hist(e_t_off, label = "$t_{off}$", alpha=0.2)
plt.hist(e_t_on, label = "$t_{on}$" , alpha=0.2)
plt.legend()
plt.show()

plt.hist(e_t_0, label = "$t_{0}$", alpha=0.2)
plt.legend()
plt.show()

f, ax = plt.subplots(2)
ax[0].scatter(e_t_0, np.array(e_t_peak)-np.array(e_t_0), c=i_sq, edgecolor='none')
ax[0].set_xlabel("$t_{0}$", size=20)
ax[0].set_ylabel("$t_{peak}$", size=20)

ax[1].scatter(e_t_on, np.array(e_t_peak)-np.array(e_t_0), c=i_sq, edgecolor='none')
ax[1].set_xlabel("$t_{On}$", size=20)
ax[1].set_ylabel("$t_{peak}$", size=20)
plt.legend()
plt.show()



#for type in neuron.experiment:
#    if type == "GABAzine":
#        for numSquares in neuron.experiment[type].keys():	
#            color = iter(plt.cm.viridis(np.linspace(0,1,24)))
#            ax = plt.subplot()
#            for coordNum in neuron.experiment[type][numSquares].coordwise:
#                t_0, t_on, t_off, g_max, sq  = [], [], [], [], []
#                c = next(color)
#                for trial in neuron.experiment[type][numSquares].coordwise[coordNum].trials:
#                    if not trial.fit == None:
#                           t_0.append(trial.fit["t_0"].value)
#                           g_max.append(trial.fit["g_max"].value)
#                           t_on.append(trial.fit["tOn"].value)
#                           t_off.append(trial.fit["tOff"].value)
#                           sq.append(numSquares)
#                ax.scatter(t_on, t_off, c=c, label=str(coordNum))
#
#            plt.xlabel("$t_{0}$", size=20)
#            plt.ylabel("$g_{max}$", size=20)
#            plt.title(str(numSquares))
#            plt.legend()
#            plt.show()
