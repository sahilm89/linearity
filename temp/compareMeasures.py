import sys
import pickle
from analysisVariables import *
import matplotlib
#matplotlib.use('cairo')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from util import plotScatterWithRegression, fitLinearRegressor
from scipy.optimize import curve_fit

controlDir = sys.argv[1] 
gabazineDir = sys.argv[2] 

class AttributeDict(dict):
    def __getattr__(self, name):
        return self[name]

nGaba = {}
oneGaba = {}
nControl = {}
oneControl = {}

gabaSlope = {}
controlSlope = {}

for measure in measureList:
    onesquarecontrolfile = controlDir + str(measure) + '_1sqr' + suffix + '.dat'
    nsquarecontrolfile = controlDir + str(measure) + '_Nsqr' + suffix + '.dat'
    
    onesquaregabazinefile = gabazineDir + str(measure) + '_1sqr' + suffix + '.dat'
    nsquaregabazinefile = gabazineDir + str(measure) + '_Nsqr' + suffix + '.dat'
    
    nsquarecontrolslopesfile = controlDir + str(measure) + '_Nsqr_slopes' + suffix + '.dat'
    nsquaregabazineslopesfile = gabazineDir + str(measure) + '_Nsqr_slopes' + suffix + '.dat'
    
    with open (onesquarecontrolfile, 'r') as infile:
        oneSquareControlData = pickle.load(infile)
    
    with open (nsquarecontrolfile, 'r') as infile:
        nSquareControlData = pickle.load(infile)
    
    with open (onesquaregabazinefile, 'r') as infile:
        oneSquareGabazineData = pickle.load(infile)
    
    with open (nsquaregabazinefile, 'r') as infile:
        nSquareGabazineData = pickle.load(infile)
    
    with open (nsquarecontrolslopesfile, 'r') as infile:
        nSquareControlSlopes = pickle.load(infile)
    
    with open (nsquaregabazineslopesfile, 'r') as infile:
        nSquareGabazineSlopes = pickle.load(infile)
    

    nGaba[measure] = AttributeDict(nSquareGabazineData)
    oneGaba[measure] = AttributeDict(oneSquareGabazineData)
    nControl[measure] = AttributeDict(nSquareControlData)
    oneControl[measure] = AttributeDict(oneSquareControlData)
    
    gabaSlope[measure] = AttributeDict(nSquareGabazineSlopes)
    controlSlope[measure] = AttributeDict(nSquareControlSlopes)


intersectingInputSquares = [val for val in nGaba[1].inputSizeofPhotoactiveGrid if val in nControl[1].inputSizeofPhotoactiveGrid]

control_timeToPeak = []
control_vmax = []
gabazine_timeToPeak = []
gabazine_vmax = []
inhibition_vmax = []

for i in intersectingInputSquares:
    for key in nGaba[1].meanOutputToInputCombs_Nsquare[i].keys():
        control_timeToPeak.append(nControl[4].meanOutputToInputCombs_Nsquare[i][key])
        control_vmax.append(nControl[1].meanOutputToInputCombs_Nsquare[i][key])
        gabazine_timeToPeak.append(nGaba[4].meanOutputToInputCombs_Nsquare[i][key])
        gabazine_vmax.append(nGaba[1].meanOutputToInputCombs_Nsquare[i][key])
        inhibition_vmax.append(nGaba[1].meanOutputToInputCombs_Nsquare[i][key] - nControl[1].meanOutputToInputCombs_Nsquare[i][key])

plt.scatter(control_vmax,control_timeToPeak, label='control',color='green')
plt.scatter(gabazine_vmax,gabazine_timeToPeak,label='gabazine',color='blue')
plt.xlabel("Vmax")
plt.ylabel("TimeToPeak")
plt.legend()
plt.savefig(controlDir + suffix + 'Vmax_TimeToPeak.' + plotType)
#plt.show()
plt.close()

print controlDir, suffix, plotType
plt.scatter(control_vmax,control_timeToPeak, label='control',c=inhibition_vmax)
plt.colorbar(label='Inhibition')
plt.xlabel("Vmax")
plt.ylabel("TimeToPeak")
plt.legend()
plt.savefig(controlDir + suffix + 'Vmax_TimeToPeak_colorbyInhibition.' + plotType)
#plt.show()
plt.close()


plt.scatter(gabazine_vmax, control_timeToPeak,  label='control')
plt.xlabel("Excitation")
plt.ylabel("Time to peak")
plt.legend()
plt.xlim((0,max(gabazine_vmax)))
plt.ylim((0.015,max(control_timeToPeak)))
plt.savefig(controlDir + suffix + 'TimeToPeak_Excitation_Vmax.' + plotType)
plt.close()
#plt.show()

plt.scatter(inhibition_vmax, control_timeToPeak, label='control')
plt.xlabel("Inhibition")
plt.ylabel("Time to peak")
plt.legend()
plt.savefig(controlDir + suffix + 'TimeToPeak_Inhibition_Vmax.' + plotType)
plt.close()
#plt.show()


############## Full lists ##########

colors = cm.viridis(np.linspace(0, 1, len(intersectingInputSquares)))

for i,c in zip(intersectingInputSquares,colors):
    control_timeToPeak = []
    control_vmax = []
    gabazine_timeToPeak = []
    gabazine_vmax = []

    control_timeToPeak.extend(nControl[4].list_Nsqr[i])
    control_vmax.extend(nControl[1].list_Nsqr[i])
    gabazine_timeToPeak.extend(nGaba[4].list_Nsqr[i])
    gabazine_vmax.extend(nGaba[1].list_Nsqr[i])
    plt.scatter(control_timeToPeak,np.log10(control_vmax), label=str(i),color=c)
#plt.scatter(gabazine_timeToPeak,np.log(gabazine_vmax), label='taug',color='green')
#plt.xlabel("time to peak")
#plt.ylabel("log-Vmax")
#plt.legend()
#plt.show()
