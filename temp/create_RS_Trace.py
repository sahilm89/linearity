#!/usr/bin/python
''' Create Artifical trace for injection '''

import matplotlib.pyplot as plt
import numpy as np

def alphaFunction(t,tau):
    ''' Returns the shape of an EPSP as an alpha function '''
    g = (t/tau)*np.exp(1-t/tau)
    return g

def doubleExponentialFunction(t, tOn,tOff):
    ''' Returns the shape of an EPSP as an double exponential function '''
    tPeak = float(((tOff * tOn)/(tOff-tOn)) * np.log(tOff/tOn))
    A = 1./(np.exp(-tPeak/tOff) - np.exp(-tPeak/tOn))
    g = A * ( np.exp(-t/tOff) - np.exp(-t/tOn))
    return g

def randomNormalVector(nrow, mu, sigma):
    '''Returns a random normal vector with given mu and sigma''' 
    vec = sigma * np.random.randn(nrow, 1) + mu
    return vec.flatten()

def createArtificial_RS_Trace(numInputs, synapseParameters, mu_amp = 0.003, sigma_amp = 0.0005, lambda_t = 0.03, points=[], acquisitionRate=20000):

    ''' This function creates artificial photostimulation data
    M = N * epsp  ## The main equation 
    '''
    signal = [] # Measurement from post-synaptic neurons
    
    if len( points):
        deltaIndices = [int(p*1e5) for p in points]
    else:
        print "blah"
        meanIndex = int(lambda_t*acquisitionRate )
        deltaIndices = np.random.poisson(meanIndex, numInputs)

    deltaT = np.array(deltaIndices,dtype=np.float)/acquisitionRate
    print deltaT

    timeIndices = np.cumsum(deltaIndices ) # Time calculated by cumulative summing on an axis 
    time = timeIndices/acquisitionRate
    t=10*lambda_t # Because epsp curves asymptote to zero.

    if len(synapseParameters) == 1:
        ##### For alpha function synapses ####################
        tau = synapseParameters[0]  ## in ms, Can also be manipulated
        epsp = alphaFunction(np.linspace(0.,t,t*acquisitionRate),tau)
    elif len(synapseParameters) == 2:
        ##### For double exponential function synapses ########
        tOn,tOff = synapseParameters  ## in ms 
        epsp = doubleExponentialFunction(np.linspace(0,t,t*acquisitionRate), tOn,tOff)
    else:
        print "wrong number of parameters entered"
        sys.exit() 

    alpha = randomNormalVector(numInputs, mu_amp, sigma_amp) # Magnitude of EPSP variability.
    sig = [np.array(epsp)*a for a in alpha]
    baseLine = np.zeros(1.2*time[-1]*acquisitionRate)
    
    for index,signal in zip(timeIndices,sig):
        baseLine[ index : index + len(signal) ] += signal 
    return baseLine 

with open('/home/sahil/Documents/Codes/bgstimPlasticity/data/deltaT_Information/Timing.txt' ,'r') as r:
    points = np.array(r.read().splitlines(),dtype=np.float)

################################ Main code ###################################
points/=7.
np.random.shuffle(points)
acquisitionRate = 20000
numInputs = 300
synapticTimeCourses = [0.002,0.005]
fileTime = 3 # In s
trace = createArtificial_RS_Trace(numInputs,synapticTimeCourses, mu_amp = 1e-1, sigma_amp = 1e-2, points = points ) # In s

newTrace = trace[:acquisitionRate*fileTime]
np.savetxt('artificialTrace_' + str(fileTime)+ '.txt', newTrace)
plt.plot(np.linspace(0,len(newTrace)/acquisitionRate,len(newTrace)), newTrace)
plt.show()
