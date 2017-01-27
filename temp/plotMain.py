import matplotlib as m
import sys
import pickle
from util import *
from analysisVariables import *

outputDir = sys.argv[1]
measure = int(sys.argv[2])

oneSquareDataFile = outputDir + str(measure) + '_1sqr' + suffix + '.dat'
nSquareDataFile = outputDir + str(measure) + '_Nsqr' + suffix + '.dat'

with open (oneSquareDataFile, 'r') as inFile:
    oneSquareData = pickle.load(inFile)

with open (nSquareDataFile, 'r') as inFile:
    nSquareData = pickle.load(inFile)

locals().update(oneSquareData)
locals().update(nSquareData)

####### Plotting here ###############################################

################# 1 square plots ################################
### Plotting heatmap of the random stimulation box here ### 
plotHeatMapBox(meanMat_1square, title= 'Mean of ' + measureDict[measure] +' of V in 1 square', measure=measureDict[measure], outputDir = outputDir, showPlots=showPlots, filename = '1_square_mean_heatmap.' + plotType, plotGridOn = sliceImage )

# Heatmap of Variance
plotHeatMapBox(var_1square, title= 'Variance ' + measureDict[measure] + ' of V: 1 square stim', measure=measureDict[measure], outputDir = outputDir, showPlots=showPlots, filename = '1_square_var_heatmap.' + plotType, plotGridOn = sliceImage)

############ Plotting histogram of measure ##############
list_1sqr = convert_1dim_DictToList(measure_1sq) # Same thing in a list
plotHistogramFromList(list_1sqr,title= "Distribution of " + measureDict[measure],outputDir=outputDir,  filename = measureDict[measure] + '_1_histogram.' + plotType, labels = [measureDict[measure], 'Frequency'], showPlots = showPlots)


################# N square plots ################################
fig_scatter_total, ax_scatter_total = plt.subplots(1,len(inputSizeofPhotoactiveGrid), figsize=(3.5*len(inputSizeofPhotoactiveGrid),3))
if len(inputSizeofPhotoactiveGrid)<2: 
    ax_scatter_total = [ax_scatter_total]

fig_scatter_total.suptitle ('Raw data plot: single square output ' + suffix)
fig_scatter_total.text(0.5, 0.04, 'single squares', ha='center')
fig_scatter_total.text(-0.04, 0.5, 'simultaneous squares', va='center', rotation='vertical')

fig_scatter_average, ax_scatter_average = plt.subplots(1,1)
fig_scatter_average.suptitle ('Normalized data plot ' + suffix)
#fig_scatter_average.text(0.5, 0.04, 'single squares', ha='center')
#fig_scatter_average.text(-0.04, 0.5, 'N normalized to one squares', va='center', rotation='vertical')

slopes = {'raw':[], 'normalized':[]}
colors = cm.viridis(np.linspace(0, 1, len(inputSizeofPhotoactiveGrid)))
for i, SizeOfPhotoactiveGrid in enumerate(inputSizeofPhotoactiveGrid):
    xlabel = "Expected sum (" + str(SizeOfPhotoactiveGrid) + ' squares)'
    ylabel = "Actual sum (" + str(SizeOfPhotoactiveGrid) + ' squares)'

    title = str(SizeOfPhotoactiveGrid) 
    fitParameters = plotScatterWithRegression(addedResponse[SizeOfPhotoactiveGrid],list_Nsqr[SizeOfPhotoactiveGrid], SizeOfPhotoactiveGrid, colorby='green',xlabel=xlabel, ylabel = ylabel, title= title, axis=ax_scatter_total[i] )
    slopes['raw'].append(fitParameters[0])

    #writeToFile(meanMat_Nsquare , outputDir + measureDict[measure] + suffix + '_' + str(SizeOfPhotoactiveGrid) +'sqr.txt')

    title = '' 
    x,y = meanMat_1square.flatten(),meanMat_Nsquare[SizeOfPhotoactiveGrid].flatten()
    fitParameters = plotScatterWithRegression(x,y, SizeOfPhotoactiveGrid, colorby=colors[i], xlabel=xlabel, ylabel = ylabel, title= title, axis=ax_scatter_average )
    slopes['normalized'].append(fitParameters[0])

    #writeToFile(fitParameters, outputDir + measureDict[measure] + suffix + '_' + str(SizeOfPhotoactiveGrid) + 'sqr_slope.txt')

with open(outputDir + str(measure) + '_' + 'Nsqr_slopes' + suffix + '.dat', 'wb') as outfile:
    pickle.dump(slopes, outfile, protocol=pickle.HIGHEST_PROTOCOL)

if (showPlots):
   fig_scatter_total.legend(fig_scatter_total.lines,fig_scatter_total.lines.labels , loc= (0.5,1.05), ncol=len(inputSizeofPhotoactiveGrid),prop={'size':6})
   fig_scatter_total.show()
else:
   fig_scatter_total.tight_layout()
   #fig_scatter_total.subplots_adjust(hspace=1.0)
   #fig_scatter_total.legend(fig_scatter_total.lines, [line.label for line in fig_scatter_total.lines], loc= (0.5,0.8), ncol=len(inputSizeofPhotoactiveGrid),prop={'size':16})
   fig_scatter_total.savefig(outputDir + measureDict[measure] + suffix + '_' + str(SizeOfPhotoactiveGrid) + '_square_domain_scatterplot.' + plotType)
   for axis in ax_scatter_total:
       x_offset = axis.xaxis.get_offset_text()
       y_offset = axis.yaxis.get_offset_text()
       x_offset_text = x_offset.get_text()
       y_offset_text = y_offset.get_text()
       x_offset.set_x(0)
       y_offset.set_x(0)
       axis.set_xlabel(u'( x {})'.format(x_offset_text),size='x-small')
       axis.set_ylabel(u'( x {})'.format(y_offset_text),size='x-small')
       x_offset.set_visible(False)
       y_offset.set_visible(False)

   fig_scatter_total.savefig(outputDir + measureDict[measure] + suffix + '_' + str(SizeOfPhotoactiveGrid) + '_square_domain_scatterplot.' + plotType)
   plt.close(fig_scatter_total)

   fig_scatter_average.tight_layout()
   #fig_scatter_total.subplots_adjust(hspace=1.0)
   #fig_scatter_average.legend(fig_scatter_total.lines, [line.label for line in fig_scatter_total.lines], loc= (0.5,0.8), ncol=len(inputSizeofPhotoactiveGrid),prop={'size':16})
   fig_scatter_average.savefig(outputDir + measureDict[measure] + suffix + '_' + str(SizeOfPhotoactiveGrid) + '_square_domain_scatterplot_average.' + plotType)
   for axis in [ax_scatter_average]:
       x_offset = axis.xaxis.get_offset_text()
       y_offset = axis.yaxis.get_offset_text()
       x_offset_text = x_offset.get_text()
       y_offset_text = y_offset.get_text()
       x_offset.set_x(0)
       y_offset.set_x(0)
       axis.set_xlabel(u'( x {})'.format(x_offset_text),size='x-small')
       axis.set_ylabel(u'( x {})'.format(y_offset_text),size='x-small')
       x_offset.set_visible(False)
       y_offset.set_visible(False)

   fig_scatter_average.savefig(outputDir + measureDict[measure] + suffix + '_' + str(SizeOfPhotoactiveGrid) + '_square_domain_scatterplot_average.' + plotType)

   plt.close(fig_scatter_average)

# All in one plot
totalInput = []
expectedTotalInput = []
inputSizeArray = []

colors = cm.viridis(np.linspace(0, 1, len(inputSizeofPhotoactiveGrid)))
max1,max2 = 0,0
fig, ax = plt.subplots()
axisWidth = (0,0)
for i,c in zip(inputSizeofPhotoactiveGrid,colors):

    axisWidth = (min(min(addedResponse[i]),min(list_Nsqr[i]),axisWidth[0]),max(max(addedResponse[i]),max(list_Nsqr[i]), axisWidth[1]))
    ax.scatter(addedResponse[i],list_Nsqr[i], color=c)
    fitParameters = fitLinearRegressor(addedResponse[i],list_Nsqr[i], domain=i, axis=ax, color=c)
    #if max1<max(addedResponse[i]):
    #    max1=max(addedResponse[i])
    #if max2<max(list_Nsqr[i]):
    #    max2=max(list_Nsqr[i])
ax.set_xlim(axisWidth)
ax.set_ylim(axisWidth)

ax.plot(axisWidth, axisWidth, 'r--' )
#plt.xlim(0,max(max1, max2))
#plt.ylim(0,max(max1, max2))
plt.xlabel("Expected sum")
plt.ylabel("Observed sum")
plt.legend(loc='upper left')

plt.savefig(outputDir + measureDict[measure] + suffix + '_all_inputSquares_scatterplot_raw.' + plotType)
plt.close()

### All in one with averaged response to a given combination of squares ##

colors = cm.viridis(np.linspace(0, 1, len(inputSizeofPhotoactiveGrid)))
max1,max2 = 0,0
fig, ax = plt.subplots()
axisWidth = (0,0)

for i,c in zip(inputSizeofPhotoactiveGrid,colors):
    expectedResponse = []
    observedResponse = []
    for key in addedResponseDict[i]:
        expectedResponse.append(addedResponseDict[i][key])
        observedResponse.append(meanOutputToInputCombs_Nsquare[i][key])
    axisWidth = (min(min(expectedResponse),min(observedResponse),axisWidth[0]),max(max(expectedResponse),max(observedResponse), axisWidth[1]))
    ax.scatter(expectedResponse,observedResponse, color=c)
    fitParameters = fitLinearRegressor(expectedResponse,observedResponse, domain=i, axis=ax, color=c)
ax.set_xlim(axisWidth)
ax.set_ylim(axisWidth)

ax.plot(axisWidth, axisWidth, 'r--' )
plt.xlabel("Expected sum")
plt.ylabel("Observed sum")
plt.legend(loc='upper left')

plt.savefig(outputDir + measureDict[measure] + suffix + '_all_inputSquares_scatterplot_raw_averaged.' + plotType)
plt.close()

# Slopes Plot
#slopeList = [slopes[inputSquares] for inputSquares in inputSizeofPhotoactiveGrid]
#productInputs = [inputSquares*slope for inputSquares, slope in zip(inputSizeofPhotoactiveGrid, slopeList)]
slopeList = copy.deepcopy(slopes['raw'])
slopeList.insert(0,1) # Adding the linear sum for indidual case
plt.plot([1] + inputSizeofPhotoactiveGrid, slopeList,'--bo', color='r', label='raw slope')

slopeList = copy.deepcopy(slopes['normalized'])
slopeList.insert(0,1) # Adding the linear sum for indidual case
#plt.plot([1] + inputSizeofPhotoactiveGrid, slopeList,'--bo', color='b', label='normalized slope')

#plt.plot([1] + inputSizeofPhotoactiveGrid,productInputs,'--bo', color='b', label='slope*input')
plt.xlabel("Input squares")
plt.ylabel("Slope value")
plt.legend()
plt.savefig(outputDir + measureDict[measure] + suffix + '_input_slopes.' + plotType)
plt.close()

######### Plotting histogram and fitting measure ########
#fitDistributionAndPlot(list_1sqr,indices = [11,12,44,58]) ## Need to put non-negative constraints and so on
#fitDistributionAndPlot(list_5sqr) ## Need to put non-negative constraints and so on

#
#
#########################################################
## Nsquare plots 
#
#    title = 'Weighted mean of ' + measureDict[measure] +' of V in '+  str(SizeOfPhotoactiveGrid) +' squares' + suffix
#    #title = 'Weighted average AOC of N square EPSP' + suffix
#    plotHeatMapBox(meanMat_Nsquare, title, measureDict[measure], outputDir = outputDir, showPlots=showPlots, filename = suffix + '_'+ str(SizeOfPhotoactiveGrid)+ '_square_mean_heatmap.' + plotType, plotGridOn= sliceImage)
#    
#    ### Plotting heatmap of the random stimulation box here ### 
#    title = 'Weighted variance ' + measureDict[measure] +' of V: ' + str(SizeOfPhotoactiveGrid) + ' square stim' + suffix
#    plotHeatMapBox(var_Nsquare, title, measureDict[measure], outputDir = outputDir, showPlots=showPlots, filename =  suffix + '_'+ str(SizeOfPhotoactiveGrid) +'_square_var_heatmap.' + plotType)
#
#    ### Plotting heatmap of the random stimulation normalized to 1sq box here ### 
#    title = 'Weighted mean of ' + measureDict[measure] +' of V normlaized to 1 sq in '+  str(SizeOfPhotoactiveGrid) +' squares' + suffix
#    residualMat = meanMat_Nsquare/meanMat_1square
#    #title = 'Weighted average AOC of N square EPSP' + suffix
#    plotHeatMapBox(residualMat, title, measureDict[measure], outputDir = outputDir, showPlots=showPlots, filename = suffix + '_'+ str(SizeOfPhotoactiveGrid)+ '_square_mean_residual_heatmap.' + plotType, plotGridOn= sliceImage)
# 
#    ############ Plotting scatter plots for comparison ##############
#    #x,y = meanMat_1square.flatten(),meanMat_Nsquare.flatten()
#    #axisWidth = (1.05*min(min(x),min(y)),1.05*max(max(x),max(y)))
#    #plt.scatter(x,y, edgecolor='blue', facecolor='none', s=40, lw='2')
#    #plt.plot(axisWidth, axisWidth, 'r--' )
#    #plt.xlim(axisWidth)
#    #plt.ylim(axisWidth)
#    #plt.xlabel('Average single sqr. stimulation')
#    #plt.ylabel('Weighted contributions for' + str(SizeOfPhotoactiveGrid) + ' sqr. stimulation')
#    #plt.annotate('Supra-linear', xy=(0.70*axisWidth[1], 0.70*axisWidth[1] ), xytext=(0.60*axisWidth[1],0.80*axisWidth[1]), arrowprops=dict(arrowstyle='<-'), ha='center', va='center')
#    #plt.annotate('Sub-linear', xy=(0.70*axisWidth[1],0.70*axisWidth[1] ), xytext=(0.80*axisWidth[1], 0.60*axisWidth[1]), arrowprops=dict(arrowstyle='<-'), ha='center', va='center')
#    #plt.suptitle('Single square vs ' + str(SizeOfPhotoactiveGrid) + ' sqr contribution' + suffix, size='large')
#    ##plt.title(measureDict[measure], size='x-large')
#    ##fitting a regression
#    #fitParameters= fitLinearRegressor(x, y, domain=1)
#    #
#    #writeToFile(fitParameters, outputDir + measureDict[measure] + suffix + '_1sqr_slope.txt')
#    #plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
#    #if (showPlots):
#    #    plt.show()
#    #else:
#    #    plt.savefig(outputDir + measureDict[measure] + suffix + '_1_square_domain_' + str(SizeOfPhotoactiveGrid) + 'sqr_scatterplot.' + plotType)
#    #plt.close()
#    #############################################################
#    #
#    ############# Plotting histograms together in a square domain for comparison #####################################
#    #
#    #plt.hist(x, bins=30, alpha = 0.75, label = '1 square', histtype='step')
#    #plt.hist(y, bins=30, alpha = 0.75, label = 'weighted ' + str(SizeOfPhotoactiveGrid) + ' square', histtype='step')
#    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#    #plt.suptitle('Distributions of 1 square and ' + str(SizeOfPhotoactiveGrid) + ' square for ' + measureDict[measure] + suffix)
#    #plt.title(str(distance_1sqr_Nsqr))
#    #
#    #if (showPlots):
#    #    plt.show()
#    #else:
#    #    plt.savefig(outputDir + measureDict[measure] +  suffix + '_' + 'histogram_1sqr_' + str(SizeOfPhotoactiveGrid) + 'sqr_weighted.' + plotType)
#    #plt.close()
#    #
#    ############################################################# 
#   
#    ############# Plotting histogram of measure ##############
#    #plt.hist(list_Nsqr,bins=50)
#    #plt.title ("Distribution of " + measureDict[measure] )
#    #plt.xlabel(measureDict[measure] + ' of voltage trace in the window of interest')
#    #plt.ylabel('Frequency')
#    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#    #if (showPlots):
#    #    plt.show()
#    #else:
#    #    plt.savefig(outputDir + measureDict[measure] + '_' + str(SizeOfPhotoactiveGrid) + '_histogram.' + plotType)
#    #plt.close()
#
#    ############ Scatter of distance with amplitude ##############
#    title = "Cluster distance vs " + measureDict[measure]
#    ylabel = measureDict[measure]
#    xlabel = 'Cluster Distance'
#    outFile = outputDir + measureDict[measure] + '_' + str(SizeOfPhotoactiveGrid) + '_spatialDistance'
#    
#    plotScatter_2_dicts(clusterDistance_Nsq, measure_Nsq, labels= (xlabel,ylabel), title = title, outFile = outFile)
#
