import numpy as np
import pickle
import matplotlib.pyplot as plt

with open ("/home/sahil/Documents/Codes/bgstimPlasticity/data/august/161013/c1/plots/c1_try.pkl",'r') as p:
    neuron = pickle.load(p)

t_0, t_on, t_off, g_max, t_peak, sq  = [], [], [], [], [], []
for type in neuron.experiment:
    if type == "GABAzine":
        for numSquares in neuron.experiment[type].keys():	
            for trialNum in neuron.experiment[type][numSquares].trial:
                if not neuron.experiment[type][numSquares].trial[trialNum].fit == None:
                    t_0.append(neuron.experiment[type][numSquares].trial[trialNum].fit["t_0"].value)
                    g_max.append(neuron.experiment[type][numSquares].trial[trialNum].fit["g_max"].value)
                    t_on.append(neuron.experiment[type][numSquares].trial[trialNum].fit["tOn"].value)
                    t_off.append(neuron.experiment[type][numSquares].trial[trialNum].fit["tOff"].value)
                    t_peak.append(neuron.experiment[type][numSquares].trial[trialNum].fit["t_peak"].value)
                     
                    sq.append(numSquares)

plt.scatter(t_off, t_0, c=sq, edgecolor='none')
plt.xlabel("$t_{off}$", size=20)
plt.ylabel("$t_{0}$", size=20)
plt.legend()
plt.show()

plt.scatter(g_max, t_0, c=sq, edgecolor='none')
plt.xlabel("$g_{max}$", size=20)
plt.ylabel("$t_{0}$", size=20)
plt.legend()
plt.show()

plt.scatter(t_off, t_on, c=sq, edgecolor='none')
plt.xlabel("$t_{off}$", size=20)
plt.ylabel("$t_{on}$", size=20)
plt.legend()
plt.show()

plt.hist(t_off, label = "$t_{off}$", alpha=0.2)
plt.hist(t_on, label = "$t_{on}$" , alpha=0.2)
plt.legend()
plt.show()

plt.hist(t_0, label = "$t_{0}$", alpha=0.2)
plt.legend()
plt.show()

f, ax = plt.subplots(2)
ax[0].scatter(t_0, np.array(t_peak)-np.array(t_0), c=sq, edgecolor='none')
ax[0].set_xlabel("$t_{0}$", size=20)
ax[0].set_ylabel("$t_{peak}$", size=20)

ax[1].scatter(t_on, np.array(t_peak)-np.array(t_0), c=sq, edgecolor='none')
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


################ Control case ##############

t_0, t_on, t_off, g_max, delta, t_0_2, t_on_2, t_off_2, g_max_2, sq, t_peak  = [], [], [], [], [], [], [], [], [], [], []
for type in neuron.experiment:
    if type == "Control":
        for numSquares in neuron.experiment[type].keys():	
            for trialNum in neuron.experiment[type][numSquares].trial:
                if not neuron.experiment[type][numSquares].trial[trialNum].fit == None:

                    t_0.append(neuron.experiment[type][numSquares].trial[trialNum].fit["t_0"].value)
                    g_max.append(neuron.experiment[type][numSquares].trial[trialNum].fit["g_max"].value)
                    t_on.append(neuron.experiment[type][numSquares].trial[trialNum].fit["tOn"].value)
                    t_off.append(neuron.experiment[type][numSquares].trial[trialNum].fit["tOff"].value)

                    sq.append(numSquares)

                    t_0_2.append(neuron.experiment[type][numSquares].trial[trialNum].fit["t_0_2"].value)
                    delta.append(neuron.experiment[type][numSquares].trial[trialNum].fit["delta"].value)
                    g_max_2.append(neuron.experiment[type][numSquares].trial[trialNum].fit["g_max_2"].value)
                    t_on_2.append(neuron.experiment[type][numSquares].trial[trialNum].fit["tOn_2"].value)
                    t_off_2.append(neuron.experiment[type][numSquares].trial[trialNum].fit["tOff_2"].value)

plt.scatter(g_max, delta, c=sq, edgecolor='none')
plt.xlabel("$g_{max}$", size=20)
plt.ylabel("$\delta$", size=20)
plt.legend()
plt.show()

plt.scatter(t_on_2, t_off_2, c=sq, edgecolor='none')
plt.xlabel("$\tau^I_{rise}$", size=20)
plt.ylabel("$\tau^I_{decay}$", size=20)
plt.legend()
plt.show()

plt.scatter(g_max, g_max_2, c=sq, edgecolor='none')
plt.xlabel("$E_{max}$", size=20)
plt.ylabel("$I_{max}$", size=20)
plt.legend()
plt.show()

plt.scatter(t_0, delta, c=sq, edgecolor='none')
plt.xlabel("$t_{0}$", size=20)
plt.ylabel("$\delta$", size=20)
plt.legend()
plt.show()
