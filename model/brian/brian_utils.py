''' This file contains all non-brian functions required for the EI balance model '''

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def simpleaxis(axes, every=False, outward=False):
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if (outward):
            ax.spines['bottom'].set_position(('outward', 10))
            ax.spines['left'].set_position(('outward', 10))
        if every:
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_title('')


def randomInput(K, N):
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr


def gridPlot(inp):
    '''Plots neurons on a grid to show active cells'''
    gridOn=True
    edgeSqr = int(np.sqrt(len(inp)))
    activeNeurons = np.reshape(inp, (edgeSqr,edgeSqr))
    cmap = LinearSegmentedColormap.from_list('CA3_reds', [(0., 'white'), (1., (170/256., 0, 0))])
    cmap.set_bad(color='white')
    fig, ax = plt.subplots()
    heatmap = ax.pcolormesh(activeNeurons,cmap=cmap)
    
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_ticks(np.arange(0,edgeSqr), minor=True)
        axis.set(ticks=np.arange(0,edgeSqr,int(edgeSqr/5))+0.5, ticklabels=np.arange(0,edgeSqr,int(edgeSqr/5))) #Skipping square labels

    if gridOn:
        ax.grid(True, which='minor', axis='both', linestyle='--', alpha=0.1, color='k')
    ax.set_xlim((0,edgeSqr))
    ax.set_ylim((0,edgeSqr))

    ax.set_aspect(1)
    fig.set_figheight(2.)
    fig.set_figwidth(2.5)
    simpleaxis(ax,every=True,outward=False)

    plt.show()


def calculateEI_imbalance(conductances, iterations):
    ''' Calculate the degree of error for this stimulation'''
    
    for j in range(iterations):
        
        ei_ratio = []
        ei_ratio.append(max(conductances[0].gi)/max(conductances[0].ge))
    ei_imbalance = (np.var(ei_ratio))
    plt.show()


def calculate_ei_delay(g_exc, g_inh):
    '''Returns the max conductance and delay'''
    return np.max(g_exc), np.flatnonzero(g_inh)[0] - np.flatnonzero(g_exc)[0]


def visualise_connectivity(S, source='Source', target='Target'):
    Ns = len(S.source)
    Nt = len(S.target)
#     figure(figsize=(10, 4))
    fig, ax = plt.subplots()
    ax.plot(np.zeros(Ns), np.arange(Ns), 'ok', ms=10)
    ax.plot(np.ones(Nt), np.arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        ax.plot([0, 1], [i, j], '-k')
    ax.set_xticks([0, 1], [source, target])
    ax.set_ylabel('Neuron index')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-1, max(Ns, Nt))
    simpleaxis(ax,every=True)
    plt.show()
