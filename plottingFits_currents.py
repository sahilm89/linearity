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
        e_t_ratio = []
        t_On_1 = []
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
                    e_t_ratio.append(neuron.experiment[type][numSquares].trial[trialNum].fit["t_ratio"].value)
                    e_sq.append(numSquares)
            for coord in neuron.experiment[type][numSquares].coordwise:
                t_o_e.append(np.average([trial.fit["t_0"] for trial in neuron.experiment[type][numSquares].coordwise[coord].trials]))
                g_max.append(np.average([trial.fit["g_max"] for trial in neuron.experiment[type][numSquares].coordwise[coord].trials]))
                t_On_1.append(np.average([trial.fit["tOn"] for trial in neuron.experiment[type][numSquares].coordwise[coord].trials]))


    if type == 2:
        i_t_0, i_t_on, i_t_off, i_g_max, i_t_peak, i_sq  = [], [], [], [], [], []
        i_t_ratio = []
        t_o_i = []
        t_On_2 = []
        for numSquares in neuron.experiment[type].keys():	
            for trialNum in neuron.experiment[type][numSquares].trial:
                print neuron.experiment[type][numSquares].trial[trialNum].fit
                if not neuron.experiment[type][numSquares].trial[trialNum].fit == None:
                    i_t_0.append(neuron.experiment[type][numSquares].trial[trialNum].fit["t_0"].value)
                    i_g_max.append(neuron.experiment[type][numSquares].trial[trialNum].fit["g_max"].value)
                    i_t_on.append(neuron.experiment[type][numSquares].trial[trialNum].fit["tOn"].value)
                    i_t_off.append(neuron.experiment[type][numSquares].trial[trialNum].fit["tOff"].value)
                    i_t_peak.append(neuron.experiment[type][numSquares].trial[trialNum].fit["t_peak"].value)
                    i_t_ratio.append(neuron.experiment[type][numSquares].trial[trialNum].fit["t_ratio"].value)
                    i_sq.append(numSquares)
            for coord in neuron.experiment[type][numSquares].coordwise:
                t_o_i.append(np.average([trial.fit["t_0"] for trial in neuron.experiment[type][numSquares].coordwise[coord].trials]))
                t_On_2.append(np.average([trial.fit["tOn"] for trial in neuron.experiment[type][numSquares].coordwise[coord].trials]))


t_o_e = np.array(t_o_e)
t_o_i = np.array(t_o_i)

delta = t_o_i - t_o_e
plt.scatter(g_max, delta)
plt.xlabel("$e_{g_max}$", size=20)
plt.ylabel("$\delta$", size=20)
plt.show()

plt.scatter(g_max, ((0.5)**1/np.array(t_On_1))* np.exp(delta/np.array(t_On_1))-((0.5)**1/np.array(t_On_2))*np.exp(delta/(np.array(t_On_2))), label="Exc/Inh")
#plt.scatter(g_max, np.array(t_On_1), label="Exc")
#plt.scatter(g_max, t_On_2, label="Inh")
plt.xlabel("$\delta$", size=20)
plt.ylabel("$tau$", size=20)
plt.legend()
plt.show()

plt.scatter(e_g_max, e_t_ratio, c=e_sq, marker = "^", label="Exc" )
plt.scatter(i_g_max, i_t_ratio, c=i_sq, marker = "o", label="Inh" )
plt.xlabel("$g_{max}$", size=20)
plt.ylabel("$t_{ratio}$", size=20)
plt.legend()
plt.colorbar()
plt.show()


plt.scatter(e_g_max, e_t_ratio, c=e_sq )
plt.xlabel("$e_{g_max}$", size=20)
plt.ylabel("$e_{tr}$", size=20)
plt.legend()
plt.colorbar()
plt.show()

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

plt.scatter(e_t_off, e_t_on, c=e_g_max, edgecolor='none')
plt.xlabel("$t_{off}$", size=20)
plt.ylabel("$t_{on}$", size=20)
plt.colorbar()
plt.legend()
plt.title("Excit")
plt.show()

plt.scatter(i_t_off, i_t_on, c=i_g_max, edgecolor='none')
plt.xlabel("$t_{off}$", size=20)
plt.ylabel("$t_{on}$", size=20)
plt.title("Inhib")
plt.colorbar()
plt.legend()
plt.show()

ax = plt.subplot()
ax.scatter(e_t_off, e_g_max, edgecolor='none', label="Off")
ax.scatter(e_t_on, e_g_max, edgecolor='none', label="On")
plt.xlabel("$taus$", size=20)
plt.ylabel("$g_{max}$", size=20)
plt.legend()
plt.title("Excit")
plt.show()

ax = plt.subplot()
ax.scatter(i_t_off, e_g_max, edgecolor='none', label="Off")
ax.scatter(i_t_on, e_g_max, edgecolor='none', label="On")
plt.xlabel("$taus$", size=20)
plt.ylabel("$g_{max}$", size=20)
plt.legend()
plt.title("Inhib")
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
