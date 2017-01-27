from util import  matrixOfZeros 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib import cm as cm

colors = [('white')] + [(cm.jet(i)) for i in xrange(1,256)]
new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)

def plotHeatMapBox(matrix, showPlots=1, filename = [], plotGridOn =[]):
    ''' Plots the heatmap of the output variable 
    in the randomStimulation box'''
    
    nrow,ncol = matrix.shape 
    alpha = 1
    column_labels = [str(i+1) for i in range(ncol)]
    row_labels = [str(i+1) for i in range(nrow)]
    
    fig, ax = plt.subplots()
    cmap=plt.cm.gray

    fmt = ScalarFormatter()
    fmt.set_powerlimits((0,0))  

    plt.gca().set_xlim((0,ncol))
    plt.gca().set_ylim((0,nrow))

    heatmap = ax.pcolor( matrix, cmap=new_map, alpha = alpha, edgecolor='black', linestyle='-', lw=2)
    #heatmap = ax.pcolor( masked_matrix, vmin= -0.000025, vmax = 0.0002, alpha = alpha)
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(nrow)+0.5, minor=False)
    ax.set_yticks(np.arange(ncol)+0.5, minor=False)
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)

    if not showPlots:
        if (filename):
            plt.savefig(filename)
        else:
            plt.savefig('heatMap.png')
    else:
        plt.show()

    plt.close()

blank = matrixOfZeros((13,13))
plotHeatMapBox(blank)
