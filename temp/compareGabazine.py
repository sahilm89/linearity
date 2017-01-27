import sys
import pickle
from analysisVariables import *
import matplotlib
#matplotlib.use('Cairo')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from util import plotScatterWithRegression, fitLinearRegressor
from scipy.optimize import curve_fit
from lmfit.models import ExponentialModel, LinearModel, ExpressionModel

controlDir = sys.argv[1] 
gabazineDir = sys.argv[2] 
measure = int(sys.argv[3])

oneSquareControlFile = controlDir + str(measure) + '_1sqr' + suffix + '.dat'
nSquareControlFile = controlDir + str(measure) + '_Nsqr' + suffix + '.dat'

oneSquareGabazineFile = gabazineDir + str(measure) + '_1sqr' + suffix + '.dat'
nSquareGabazineFile = gabazineDir + str(measure) + '_Nsqr' + suffix + '.dat'

nSquareControlSlopesFile = controlDir + str(measure) + '_Nsqr_slopes' + suffix + '.dat'
nSquareGabazineSlopesFile = gabazineDir + str(measure) + '_Nsqr_slopes' + suffix + '.dat'

with open (oneSquareControlFile, 'r') as inFile:
    oneSquareControlData = pickle.load(inFile)

with open (nSquareControlFile, 'r') as inFile:
    nSquareControlData = pickle.load(inFile)

with open (oneSquareGabazineFile, 'r') as inFile:
    oneSquareGabazineData = pickle.load(inFile)

with open (nSquareGabazineFile, 'r') as inFile:
    nSquareGabazineData = pickle.load(inFile)

with open (nSquareControlSlopesFile, 'r') as inFile:
    nSquareControlSlopes = pickle.load(inFile)

with open (nSquareGabazineSlopesFile, 'r') as inFile:
    nSquareGabazineSlopes = pickle.load(inFile)

class AttributeDict(dict):
    def __getattr__(self, name):
        return self[name]

nGaba = AttributeDict(nSquareGabazineData)
oneGaba = AttributeDict(oneSquareGabazineData)
nControl = AttributeDict(nSquareControlData)
oneControl = AttributeDict(oneSquareControlData)

gabaSlope = AttributeDict(nSquareGabazineSlopes)
controlSlope = AttributeDict(nSquareControlSlopes)

intersectingInputSquares = [val for val in nGaba.inputSizeofPhotoactiveGrid if val in nControl.inputSizeofPhotoactiveGrid]

print nGaba.inputSizeofPhotoactiveGrid
print nControl.inputSizeofPhotoactiveGrid
print gabaSlope.raw
print gabaSlope.normalized
print controlSlope.raw
print controlSlope.normalized

###### Taking differentials of the data 
gabazine_observed = []
control_observed = []
gabazine_expected = []
control_expected = []
inputSquares_array = []

#for key in oneGaba.meanOutputToInputCombs_1square.keys():
#    if key in oneControl.meanOutputToInputCombs_1square:
#        control_observed.append(oneControl.meanOutputToInputCombs_1square[key])
#        control_expected.append(oneControl.meanOutputToInputCombs_1square[key])
#        gabazine_observed.append(oneGaba.meanOutputToInputCombs_1square[key])
#        gabazine_expected.append(oneGaba.meanOutputToInputCombs_1square[key])
#        inputSquares_array.append(1)

for i in intersectingInputSquares:
    for key in nGaba.meanOutputToInputCombs_Nsquare[i]:
        if key in nControl.meanOutputToInputCombs_Nsquare[i]:
            control_observed.append(nControl.meanOutputToInputCombs_Nsquare[i][key])
            control_expected.append(nControl.addedResponseDict[i][key])
            gabazine_observed.append(nGaba.meanOutputToInputCombs_Nsquare[i][key])
            gabazine_expected.append(nGaba.addedResponseDict[i][key])
            inputSquares_array.append(i)

control_expected, control_observed, gabazine_expected, gabazine_observed,inputSquares_array = zip(*sorted(zip(control_expected, control_observed, gabazine_expected, gabazine_observed,inputSquares_array)))

control_expected_prime, control_observed_prime, gabazine_expected_prime, gabazine_observed_prime = np.diff(control_expected), np.diff(control_observed), np.diff(gabazine_expected), np.diff(gabazine_observed)

colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(intersectingInputSquares)))
cs = []
for inputSquare in inputSquares_array:
    for i,x in enumerate(intersectingInputSquares):
        if inputSquare == x:
            cs.append(colors[i])
print inputSquares_array
print cs
fig, ax = plt.subplots()
#ax.scatter(control_expected[:-1],control_observed_prime/control_expected_prime, c=cs[:-1],marker = "^", label="control")
ax.plot(control_expected[:-1],control_observed_prime/control_expected_prime, 'b--',label="control")
ax.plot(gabazine_expected[:-1],gabazine_observed_prime/gabazine_expected_prime, 'r--',label="gabazine")

gabazine_expected, gabazine_observed = zip(*sorted(zip(gabazine_expected, gabazine_observed)))
gabazine_expected_prime, gabazine_observed_prime = np.diff(gabazine_expected), np.diff(gabazine_observed)

ax.plot(gabazine_expected[:-1],gabazine_observed_prime/gabazine_expected_prime, 'g--',label="gabazine")

ax.legend()
#ax.scatter(gabazine_expected[:-1],gabazine_observed_prime/gabazine_expected_prime, color='b',marker = ".", label="gabazine")
plt.show()

plt.scatter(control_expected,np.exp((np.array(control_observed)-control_expected)/1.), c='g')
plt.scatter(gabazine_expected,np.exp((np.array(gabazine_observed)-gabazine_expected)/1.), c='r')
plt.show()

#
## Slopes Plot
#slopeList = [slopes[inputSquares] for inputSquares in nGaba.inputSizeofPhotoactiveGrid]
#productInputs = [inputSquares*slope for inputSquares, slope in zip(inputSizeofPhotoactiveGrid, slopeList)]
#slopeList.insert(0,1) # Adding the linear sum for indidual case
plt.plot([1] + nControl.inputSizeofPhotoactiveGrid,[1.] + controlSlope.normalized ,'--bo', color='b', label='Normalized_Control')
plt.plot([1] + nGaba.inputSizeofPhotoactiveGrid,[1.] + gabaSlope.normalized ,'--bo', color='g', label='Normalized_Gabazine')
#plt.plot([1] + inputSizeofPhotoactiveGrid,productInputs,'--bo', color='b', label='slope*input')
plt.xlabel("Input squares")
plt.ylabel("Slope value")
plt.legend()
plt.savefig(controlDir + measureList[measure] + suffix + '_input_slopes_normalized.' + plotType)
#plt.show()
plt.close()

############ Divisive normalization for slope inputs #############################
inverseGain = [1./slopeValue for slopeValue in nGaba.inputSizeofPhotoactiveGrid]
inverseExpGain = [np.exp(-slopeValue) for slopeValue in nGaba.inputSizeofPhotoactiveGrid]

plt.plot([1] + nControl.inputSizeofPhotoactiveGrid,[1.] + controlSlope.raw ,'-bo', color='b', label='Control')
plt.plot([1] + nGaba.inputSizeofPhotoactiveGrid,[1.] + gabaSlope.raw ,'-bo', color='g', label='Gabazine')
plt.plot([1] + nGaba.inputSizeofPhotoactiveGrid, [1.] + inverseGain ,'--', color='r',label='Perfect division')
plt.plot([1] + nGaba.inputSizeofPhotoactiveGrid, [1.] + inverseExpGain ,'--', color='orange',label='Perfect exponentiation')
#plt.plot([1] + inputSizeofPhotoactiveGrid,productInputs,'--bo', color='b', label='slope*input')
plt.xlabel("Input squares")
plt.ylabel("Slope value")
plt.legend()
plt.savefig(controlDir + measureList[measure] + suffix + '_input_slopes_control.' + plotType)
#plt.show()
plt.close()




############ Divisive normalization for slope inputs: Only control #############################
inverseGain = [1./slopeValue for slopeValue in nGaba.inputSizeofPhotoactiveGrid]
inverseExpGain = [np.exp(-slopeValue) for slopeValue in nGaba.inputSizeofPhotoactiveGrid]

plt.plot([1] + nControl.inputSizeofPhotoactiveGrid,[1.] + controlSlope.raw ,'-bo', color='b', label='Control')
plt.plot([1] + nGaba.inputSizeofPhotoactiveGrid, [1.] + inverseGain ,'--', color='r',label='Perfect division')
plt.plot([1] + nGaba.inputSizeofPhotoactiveGrid, [1.] + inverseExpGain ,'--', color='orange',label='Perfect exponentiation')
#plt.plot([1] + inputSizeofPhotoactiveGrid,productInputs,'--bo', color='b', label='slope*input')
plt.xlabel("Input squares")
plt.ylabel("Slope value")
plt.legend()
plt.savefig(controlDir + measureList[measure] + suffix + '_input_slopes_control_only.' + plotType)
#plt.show()
plt.close()



###### Subtract Gabazine from control ###################
#diffGabaControl_Nsquare = {}
#assert nGaba.inputSizeofPhotoactiveGrid == nControl.inputSizeofPhotoactiveGrid, "Mismatch in number of input squares."
#
#diffGabaControl_1square =  oneGaba.meanMat_1square - oneControl.meanMat_1square
#for key in sorted(nGaba.meanMat_Nsquare.keys()):
#    diffGabaControl_Nsquare[key] =  nGaba.meanMat_Nsquare[key] - nControl.meanMat_Nsquare[key]
#
#colors = cm.viridis(np.linspace(0, 1, len(nGaba.inputSizeofPhotoactiveGrid)))
#for i, SizeOfPhotoactiveGrid in enumerate(nGaba.inputSizeofPhotoactiveGrid):
#    xlabel = "Expected sum (" + str(SizeOfPhotoactiveGrid) + ' squares)'
#    ylabel = "Actual sum (" + str(SizeOfPhotoactiveGrid) + ' squares)'
#    plt.scatter(diffGabaControl_1square,diffGabaControl_Nsquare[SizeOfPhotoactiveGrid], c=colors[i],label=str(SizeOfPhotoactiveGrid))
#    plt.xlabel(xlabel)
#    plt.ylabel(ylabel)
#
#    #fitParameters = plotScatterWithRegression(diffGabaControl_1square,diffGabaControl_Nsquare[SizeOfPhotoactiveGrid], SizeOfPhotoactiveGrid, colorby='green',xlabel=xlabel, ylabel = ylabel)
#plt.legend()
#plt.show()

#oneGaba = AttributeDict(oneSquareGabazineData)
#nControl = AttributeDict(nSquareControlData)
#oneControl = AttributeDict(oneSquareControlData)


### Divisive normalization (linearized plot) for expected EPSP amplitudes and Gabazine ############

#colors = cm.viridis(np.linspace(0, 1, len(intersectingInputSquares)))
#max1,max2 = 0,0
#fig, ax = plt.subplots()
#axisWidth_x = (0,0)
#axisWidth_y = (0,0)
#
#expectedResponse_control = []
#response_ratio_control = []
#expectedResponse_gabazine = []
#observedResponse_gabazine = []
#
#response_ratio_gabazine = []
#excitation_expected_ratio = []
#inhibition_expected_ratio = []
#inh_exc_expected_ratio = []
#
#for i,c in zip(intersectingInputSquares,colors):
#    for key in nGaba.meanOutputToInputCombs_Nsquare[i]:
#        if key in nControl.meanOutputToInputCombs_Nsquare[i]:
#            expectedResponse_control.append(nControl.addedResponseDict[i][key])
#            expectedResponse_gabazine.append(nGaba.addedResponseDict[i][key])
#            observedResponse_gabazine.append(nGaba.meanOutputToInputCombs_Nsquare[i][key])
#            response_ratio_control.append(nControl.addedResponseDict[i][key]/nControl.meanOutputToInputCombs_Nsquare[i][key])
#            response_ratio_gabazine.append(nGaba.addedResponseDict[i][key]/nGaba.meanOutputToInputCombs_Nsquare[i][key])
#            excitation_expected_ratio.append((nGaba.meanOutputToInputCombs_Nsquare[i][key])/nGaba.addedResponseDict[i][key])
#            inhibition_expected_ratio.append((nGaba.meanOutputToInputCombs_Nsquare[i][key]-nControl.meanOutputToInputCombs_Nsquare[i][key])/nGaba.addedResponseDict[i][key])
#            inh_exc_expected_ratio.append((nGaba.meanOutputToInputCombs_Nsquare[i][key]/nControl.meanOutputToInputCombs_Nsquare[i][key]))
#
#axisWidth_x = (min(min(expectedResponse_control),min(expectedResponse_gabazine)),max(max(expectedResponse_control),max(expectedResponse_gabazine)))
#axisWidth_y = (min(min(response_ratio_control),min(response_ratio_gabazine)),max(max(response_ratio_control),max(response_ratio_gabazine)))
#
#ax.scatter(expectedResponse_control,response_ratio_control,color='g',marker = "^", label="control ")
#ax.scatter(observedResponse_gabazine,inh_exc_expected_ratio, color='b',marker = ".", label="inhibition only ")
##ax.scatter(expectedResponse_gabazine,excitation_expected_ratio, color=c,marker = "^", label="excitation only " + str(i))
##ax.scatter(expectedResponse_gabazine,inhibition_expected_ratio, color=c,marker = "o", label="inhibition only " + str(i))
##ax.scatter(expectedResponse_gabazine,response_ratio_gabazine, color=c, label="gabazine " + str(i))
#
##popt, pcov = curve_fit(division, expectedResponse_control, response_ratio_control)
##inverseGain = [division(x,popt) for x in np.linspace(axisWidth_x[0], axisWidth_x[1], 100)]
##ax.plot(np.linspace(axisWidth_x[0], axisWidth_x[1], 100), inverseGain ,'--', color='r',label='k1/x, k1={:.4f}'.format( popt[0]))
##
##
##popt, pcov = curve_fit(division, observedResponse_gabazine,inh_exc_expected_ratio)
##inverseGain = [division(x,popt) for x in np.linspace(axisWidth_x[0], axisWidth_x[1], 100)]
##ax.plot(np.linspace(axisWidth_x[0], axisWidth_x[1], 100), inverseGain ,'--', color='g',label='k2/x, k2={:.4f}'.format( popt[0]))
#
#fitLinearRegressor(expectedResponse_control, response_ratio_control,color='g')
#fitLinearRegressor(observedResponse_gabazine, inh_exc_expected_ratio,color='b')
##
#ax.set_xlim(axisWidth_x)
#ax.set_ylim(axisWidth_y)
#
##ax.plot(axisWidth, axisWidth, 'r--' )
#plt.xlabel("Expected Response and Gabazine Observed")
#plt.ylabel("Expected/Observed and Gabazine/Control")
#plt.legend(scatterpoints = 1, loc='upper left')
#plt.show()
##plt.savefig(controlDir + measureList[measure] + suffix + '_Divisive_normalization_Expected_ratio_OE_control_and_Gabazine_ratio_CG.' + plotType)
#plt.close()
#
#
#### Divisive normalization for expected EPSP amplitudes and Gabazine ############
#def division(x, k):
#    return float(k)/x 
#
#def exponentiation(x,k,m):
#    return m*np.log(k*x)
#
#
#colors = cm.viridis(np.linspace(0, 1, len(intersectingInputSquares)))
#max1,max2 = 0,0
#fig, ax = plt.subplots()
#axisWidth_x = (0,0)
#axisWidth_y = (0,0)
#
#expectedResponse_control = []
#response_ratio_control = []
#expectedResponse_gabazine = []
#observedResponse_gabazine = []
#
#response_ratio_gabazine = []
#excitation_expected_ratio = []
#inhibition_expected_ratio = []
#exc_inh_expected_ratio = []
#
#for i,c in zip(intersectingInputSquares,colors):
#    for key in nGaba.meanOutputToInputCombs_Nsquare[i]:
#        if key in nControl.meanOutputToInputCombs_Nsquare[i]:
#            expectedResponse_control.append(nControl.addedResponseDict[i][key])
#            expectedResponse_gabazine.append(nGaba.addedResponseDict[i][key])
#            observedResponse_gabazine.append(nGaba.meanOutputToInputCombs_Nsquare[i][key])
#            response_ratio_control.append(nControl.meanOutputToInputCombs_Nsquare[i][key]/nControl.addedResponseDict[i][key])
#            response_ratio_gabazine.append(nGaba.meanOutputToInputCombs_Nsquare[i][key]/nGaba.addedResponseDict[i][key])
#            excitation_expected_ratio.append((nGaba.meanOutputToInputCombs_Nsquare[i][key])/nGaba.addedResponseDict[i][key])
#            inhibition_expected_ratio.append((nGaba.meanOutputToInputCombs_Nsquare[i][key]-nControl.meanOutputToInputCombs_Nsquare[i][key])/nGaba.addedResponseDict[i][key])
#            exc_inh_expected_ratio.append(nControl.meanOutputToInputCombs_Nsquare[i][key]/nGaba.meanOutputToInputCombs_Nsquare[i][key])
#
#axisWidth_x = (min(min(expectedResponse_control),min(expectedResponse_gabazine)),max(max(expectedResponse_control),max(expectedResponse_gabazine)))
#axisWidth_y = (min(min(response_ratio_control),min(response_ratio_gabazine)),max(max(response_ratio_control),max(response_ratio_gabazine)))
#
#ax.scatter(expectedResponse_control,response_ratio_control,color='g',marker = "^", label="control ")
##ax.scatter(observedResponse_gabazine,exc_inh_expected_ratio, color='b',marker = ".", label="excitation only (Gabazine) ")
##ax.scatter(expectedResponse_gabazine,excitation_expected_ratio, color=c,marker = "^", label="excitation only " + str(i))
##ax.scatter(expectedResponse_gabazine,inhibition_expected_ratio, color=c,marker = "o", label="inhibition only " + str(i))
##ax.scatter(expectedResponse_gabazine,response_ratio_gabazine, color=c, label="gabazine " + str(i))
#
#popt, pcov = curve_fit(division, expectedResponse_control, response_ratio_control)
#inverseGain = [division(x,popt) for x in np.linspace(axisWidth_x[0], axisWidth_x[1], 100)]
#ax.plot(np.linspace(axisWidth_x[0], axisWidth_x[1], 100), inverseGain ,'--', color='r',label='k1/x, k1={:.4f}'.format( popt[0]))
#
#popt, pcov = curve_fit(exponentiation, expectedResponse_control, response_ratio_control)
#inverseGain = [exponentiation(x,popt[0],popt[1]) for x in np.linspace(axisWidth_x[0], axisWidth_x[1], 100)]
#ax.plot(np.linspace(axisWidth_x[0], axisWidth_x[1], 100), inverseGain ,'--', color='g',label='ae^kx, k={:.4f}'.format( popt[0]))
#
##
##
##
##popt, pcov = curve_fit(division, observedResponse_gabazine,exc_inh_expected_ratio)
##inverseGain = [division(x,popt) for x in np.linspace(axisWidth_x[0], axisWidth_x[1], 100)]
##ax.plot(np.linspace(axisWidth_x[0], axisWidth_x[1], 100), inverseGain ,'--', color='g',label='k2/x, k2={:.4f}'.format( popt[0]))
#
##
#ax.set_xlim(axisWidth_x)
#ax.set_ylim(axisWidth_y)
#
##ax.plot(axisWidth, axisWidth, 'r--' )
#plt.xlabel("Expected Response")# and Gabazine Observed")
#plt.ylabel("Observed/Expected")# and Control/Gabazine")
#plt.legend(scatterpoints = 1, loc='upper right')
#plt.show()
##plt.savefig(controlDir + measureList[measure] + suffix + '_Divisive_normalization_Expected_ratio_OE_control_and_Gabazine_ratio_CG.' + plotType)
#plt.close()


##### Subtract Gabazine from control, averaged responses against Expected Control  ###################
max1,max2 = 0,0
fig, ax = plt.subplots()
axisWidth = (0,0)

#assert nGaba.inputSizeofPhotoactiveGrid == nControl.inputSizeofPhotoactiveGrid, "Mismatch in number of input squares."
for i,c in zip(intersectingInputSquares,colors):
    observedEPSP = []
    observedResponse = []
    for key in nGaba.meanOutputToInputCombs_Nsquare[i]:
        if key in nControl.meanOutputToInputCombs_Nsquare[i]:
            observedEPSP.append(nGaba.meanOutputToInputCombs_Nsquare[i][key])
            observedResponse.append((nGaba.meanOutputToInputCombs_Nsquare[i][key]-nControl.meanOutputToInputCombs_Nsquare[i][key])/nGaba.meanOutputToInputCombs_Nsquare[i][key])

    ax.scatter(observedEPSP,observedResponse, color=c)
    #fitParameters = fitLinearRegressor(observedEPSP,observedResponse, domain=i, axis=ax, color=c)

plt.xlabel("Excitation (Observed)")
plt.ylabel("Specific Inhibition")
plt.legend()

plt.show()
#plt.savefig(controlDir + measureList[measure] + suffix + '_specific_inhibition_Excitation_raw.' + plotType)
plt.close()
#
#
###### Relative inhibition plot
#max1,max2 = 0,0
#fig, ax = plt.subplots()
#axisWidth = (0,0)
#
#
#for i,c in zip(nGaba.inputSizeofPhotoactiveGrid,colors):
#    observedEPSP = []
#    observedResponse = []
#    for key in nGaba.meanOutputToInputCombs_Nsquare[i]:
#        if key in nControl.meanOutputToInputCombs_Nsquare[i]:
#            observedEPSP.append((nGaba.addedResponseDict[i][key]-nControl.addedResponseDict[i][key])/nControl.addedResponseDict[i][key])
#            observedResponse.append((nGaba.meanOutputToInputCombs_Nsquare[i][key]-nControl.meanOutputToInputCombs_Nsquare[i][key])/nControl.meanOutputToInputCombs_Nsquare[i][key])
#
#    axisWidth = (min(min(observedEPSP),min(observedResponse),axisWidth[0]),max(max(observedEPSP),max(observedResponse), axisWidth[1]))
#    ax.scatter(observedEPSP,observedResponse, color=c)
#    fitParameters = fitLinearRegressor(observedEPSP,observedResponse, domain=i, axis=ax, color=c)
#ax.set_xlim(axisWidth)
#ax.set_ylim(axisWidth)
#
#ax.plot(axisWidth, axisWidth, 'r--' )
#plt.xlabel("Relative inhibition (Expected)")
#plt.ylabel("Relative inhibition (Observed)")
#plt.legend()
#plt.title("Relative Inhibition")
#plt.show()
##plt.savefig(outputDir + measureList[measure] + suffix + '_all_inputSquares_scatterplot_raw.' + plotType)
#plt.close()
#

###### Subtract Gabazine from control, averaged responses against Squares ###################
max1,max2 = 0,0
fig, ax = plt.subplots()
axisWidth = (0,0)

observedResponse= []

# One square relative inhibition 

observedResponse_Square = [(oneGaba.meanOutputToInputCombs_1square[key]-oneControl.meanOutputToInputCombs_1square[key])/oneGaba.meanOutputToInputCombs_1square[key] for key in oneGaba.meanOutputToInputCombs_1square.keys()]

observedResponse.append(observedResponse_Square)

for i,c in zip(intersectingInputSquares,colors):
    observedResponse_Square = []
    for key in nGaba.meanOutputToInputCombs_Nsquare[i]:
        if key in nControl.meanOutputToInputCombs_Nsquare[i]:
            observedResponse_Square.append((nGaba.meanOutputToInputCombs_Nsquare[i][key]-nControl.meanOutputToInputCombs_Nsquare[i][key])/nGaba.meanOutputToInputCombs_Nsquare[i][key])
    observedResponse.append(observedResponse_Square)

            #axisWidth = (min(min(observedResponse_Square),axisWidth[0]),max(max(observedResponse), axisWidth[1]))
    #ax.scatter([i]*len(observedResponse),observedResponse, color=c)
print len(observedResponse), intersectingInputSquares 
ax.violinplot(observedResponse, [1] + intersectingInputSquares, points=20, widths=1.0, showmeans=True, showextrema=True, showmedians=True, bw_method='silverman')

#ax.set_ylim(axisWidth,)
#ax.set_xlim((0,10))

plt.xlabel("Number of input squares ")
plt.ylabel("Relative Inhibition")
plt.legend()

#plt.show()
plt.savefig(controlDir + measureList[measure] + suffix + '_all_inputSquares_violinplot_raw_averaged.' + plotType)
plt.close()

########## E-I plot ##########################################

max1,max2 = 0,0
fig, ax = plt.subplots()
axisWidth = (0,0)

###### One square 
excitation = []
inhibition = []

for key in oneGaba.meanOutputToInputCombs_1square.keys():
    if key in oneControl.meanOutputToInputCombs_1square:
        inhibition.append(oneGaba.meanOutputToInputCombs_1square[key]-oneControl.meanOutputToInputCombs_1square[key])
        excitation.append(oneGaba.meanOutputToInputCombs_1square[key])

axisWidth = (min(min(excitation),min(inhibition),axisWidth[0]),max(max(excitation),max(inhibition), axisWidth[1]))
ax.scatter(excitation,inhibition, color='black', label='1')
 
for i,c in zip(intersectingInputSquares,colors):
    excitation = []
    inhibition = []
    for key in nGaba.meanOutputToInputCombs_Nsquare[i]:
        if key in nControl.meanOutputToInputCombs_Nsquare[i]:
            inhibition.append(nGaba.meanOutputToInputCombs_Nsquare[i][key]-nControl.meanOutputToInputCombs_Nsquare[i][key])
            excitation.append(nGaba.meanOutputToInputCombs_Nsquare[i][key])

    axisWidth = (min(min(excitation),min(inhibition),axisWidth[0]),max(max(excitation),max(inhibition), axisWidth[1]))
    ax.scatter(excitation,inhibition, color=c, label=str(i))
    #fitParameters = fitLinearRegressor(excitation,inhibition, domain=i, axis=ax, color=c)

ax.set_xlim(axisWidth)
ax.set_ylim(axisWidth)
ax.plot(axisWidth, axisWidth, 'r--' )

plt.xlabel("Total Excitation")
plt.ylabel("Total Inhibition")
plt.title("Excitation-Inhibition balance for all inputs to a cell")
plt.legend(loc='upper left')

#plt.show()
plt.savefig(controlDir + measureList[measure] + suffix + '_Excitation_Inhibition_scatterplot_raw.' + plotType)
plt.close()

################### Observed Control versus Observed Gabazine ############
max1,max2 = 0,0
fig, ax = plt.subplots()
axisWidth = (0,0)

###### One square 
gabazine = []
control = []

for key in oneGaba.meanOutputToInputCombs_1square.keys():
    if key in oneControl.meanOutputToInputCombs_1square:
        gabazine.append(oneGaba.meanOutputToInputCombs_1square[key])
        control.append(oneControl.meanOutputToInputCombs_1square[key])

axisWidth = (min(min(gabazine),min(control),axisWidth[0]),max(max(gabazine),max(control), axisWidth[1]))
ax.scatter(gabazine,control, color='black', label='1')
 
for i,c in zip(intersectingInputSquares,colors):
    gabazine = []
    control = []
    for key in nGaba.meanOutputToInputCombs_Nsquare[i]:
        if key in nControl.meanOutputToInputCombs_Nsquare[i]:
            control.append(nControl.meanOutputToInputCombs_Nsquare[i][key])
            gabazine.append(nGaba.meanOutputToInputCombs_Nsquare[i][key])

    axisWidth = (min(min(gabazine),min(control),axisWidth[0]),max(max(gabazine),max(control), axisWidth[1]))
    ax.scatter(gabazine,control, color=c, label=str(i))
    #fitParameters = fitLinearRegressor(gabazine,control, domain=i, axis=ax, color=c)

ax.set_xlim(axisWidth)
ax.set_ylim(axisWidth)
ax.plot(axisWidth, axisWidth, 'r--' )

plt.xlabel("Observed Gabazine")
plt.ylabel("Observed Control")
plt.title("gabazine-control balance for all inputs to a cell")
plt.legend(loc='upper left')

#plt.show()
plt.savefig(controlDir + measureList[measure] + suffix + '_gabazine_control_scatterplot_raw.' + plotType)
plt.close()

###### Hao et al metric : Sum = E + I + k.E.I
#### f(e) = (theta - e )/(1+k*e)

#colors = cm.viridis(np.linspace(0, 1, len(intersectingInputSquares)))
#max1,max2 = 0,0
#fig, ax = plt.subplots()
#axisWidth_x = (0,0)
#axisWidth_y = (0,0)
#
#expectedResponse_control = []
#observedResponse_control = []
#response_ratio_control = []
#expectedResponse_gabazine = []
#observedResponse_gabazine = []
#
#response_ratio_gabazine = []
#excitation_expected_ratio = []
#inhibition_expected_ratio = []
#inh_exc_expected_ratio = []
#
#for i,c in zip(intersectingInputSquares,colors):
#    for key in nGaba.meanOutputToInputCombs_Nsquare[i]:
#        if key in nControl.meanOutputToInputCombs_Nsquare[i]:
#            observedResponse_control.append(nControl.meanOutputToInputCombs_Nsquare[i][key])
#            observedResponse_gabazine.append(nGaba.meanOutputToInputCombs_Nsquare[i][key])
#            response_ratio_gabazine.append(nGaba.addedResponseDict[i][key]/nGaba.meanOutputToInputCombs_Nsquare[i][key])
#            excitation_expected_ratio.append((nGaba.meanOutputToInputCombs_Nsquare[i][key])/nGaba.addedResponseDict[i][key])
#            inhibition_expected_ratio.append((nGaba.meanOutputToInputCombs_Nsquare[i][key]-nControl.meanOutputToInputCombs_Nsquare[i][key])/nGaba.addedResponseDict[i][key])
#            inh_exc_expected_ratio.append((nGaba.meanOutputToInputCombs_Nsquare[i][key]/nControl.meanOutputToInputCombs_Nsquare[i][key]))
#
#axisWidth_x = (min(min(expectedResponse_control),min(expectedResponse_gabazine)),max(max(expectedResponse_control),max(expectedResponse_gabazine)))
#axisWidth_y = (min(min(response_ratio_control),min(response_ratio_gabazine)),max(max(response_ratio_control),max(response_ratio_gabazine)))
#
#ax.scatter(expectedResponse_control,response_ratio_control,color='g',marker = "^", label="control ")
#ax.scatter(observedResponse_gabazine,inh_exc_expected_ratio, color='b',marker = ".", label="inhibition only ")
##ax.scatter(expectedResponse_gabazine,excitation_expected_ratio, color=c,marker = "^", label="excitation only " + str(i))
##ax.scatter(expectedResponse_gabazine,inhibition_expected_ratio, color=c,marker = "o", label="inhibition only " + str(i))
##ax.scatter(expectedResponse_gabazine,response_ratio_gabazine, color=c, label="gabazine " + str(i))
#
##popt, pcov = curve_fit(division, expectedResponse_control, response_ratio_control)
##inverseGain = [division(x,popt) for x in np.linspace(axisWidth_x[0], axisWidth_x[1], 100)]
##ax.plot(np.linspace(axisWidth_x[0], axisWidth_x[1], 100), inverseGain ,'--', color='r',label='k1/x, k1={:.4f}'.format( popt[0]))
##
##
##popt, pcov = curve_fit(division, observedResponse_gabazine,inh_exc_expected_ratio)
##inverseGain = [division(x,popt) for x in np.linspace(axisWidth_x[0], axisWidth_x[1], 100)]
##ax.plot(np.linspace(axisWidth_x[0], axisWidth_x[1], 100), inverseGain ,'--', color='g',label='k2/x, k2={:.4f}'.format( popt[0]))
#
#fitLinearRegressor(expectedResponse_control, response_ratio_control,color='g')
#fitLinearRegressor(observedResponse_gabazine, inh_exc_expected_ratio,color='b')
##
#ax.set_xlim(axisWidth_x)
#ax.set_ylim(axisWidth_y)
#
##ax.plot(axisWidth, axisWidth, 'r--' )
#plt.xlabel("Expected Response and Gabazine Observed")
#plt.ylabel("Expected/Observed and Gabazine/Control")
#plt.legend(scatterpoints = 1, loc='upper left')
##plt.show()
#plt.savefig(controlDir + measureList[measure] + suffix + '_Divisive_normalization_Expected_ratio_OE_control_and_Gabazine_ratio_CG.' + plotType)
#plt.close()

exp_mod = ExponentialModel(prefix='exp_')
line_mod = LinearModel(prefix='line_')
#pars = mod.guess(control_observed, x=control_expected)
#print pars

pars = exp_mod.guess(control_observed, x=control_expected)
pars += line_mod.guess(control_observed, x=control_expected)
#pars +=  line_mod.make_params(intercept=np.min(control_observed), slope=0)

mod = exp_mod + line_mod
out = mod.fit(control_observed, pars, x=control_expected)

print(out.fit_report())

plt.scatter(control_expected, control_observed)
plt.plot(control_expected, out.init_fit, 'k--')
plt.plot(control_expected, out.best_fit, 'r-')
plt.show()

mod = ExpressionModel('b + c*log(x)')
#pars = mod.guess(control_observed, x=control_expected)
params = mod.make_params(b=0, c=1)
out = mod.fit(control_observed, params, x=control_expected)

print(out.fit_report())

plt.scatter(control_expected, control_observed)
#plt.plot(control_expected, out.init_fit, 'k--')
plt.plot(control_expected, out.best_fit, 'r-')
plt.show()
