#!/usr/bin/python
'''
Each synapse is a binomial process with certain probability p.
Each neuron sums up scaled binomials at soma.
'''

import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    ''' Neuron class for the small simulation '''

    def __init__(self, numSynapses=100, releaseP=0.2, distribution='normal', *parms):
        '''Initialize neuron class''' 
        self.numSynapses = numSynapses
        self.synapticWeightDistribution = self.generate_synaptic_weight_distribution(distribution, *parms)

        assert 0.<=releaseP<=1., "Release probability must be between 0 and 1"
        self.releaseP = releaseP

        self.synapticInput = np.random.binomial(1,self.releaseP, self.numSynapses)
        self.synapticInput = self.__scaleInput__(self.synapticInput, self.synapticWeightDistribution) 

    def __scaleInput__(self, synapticInput, scalingDistribution):
        ''' Scales the binomial synaptic input with synapti_c weight distributions '''
        return synapticInput * scalingDistribution 

    def generate_synaptic_weight_distribution(self, distribution='normal', *distPars): 
        ''' Returns the synaptic weight distribution for the neuron '''
        if distribution == 'normal':
            assert len(distPars)==2, "Incorrect number of parameters for a normal distribution."
            return np.random.normal(distPars[0],distPars[1],self.numSynapses)
        elif distribution == 'poisson':
            assert len(distPars)==1, "Incorrect number of parameters for a poisson distribution."
            return np.random.poisson(distPars[0],self.numSynapses)
        else:
            raise "Distribution not recognized"

    def sumInputs(self):
        return np.sum(self.synapticInput)

totalSum = []
N_array = np.logspace(0,10,num=10,base=2,dtype=np.int)

for N in N_array:
    neuron1 = Neuron(N,.2, 'poisson', 1)
    neuron1.releaseP = 0.5
    totalSum.append(neuron1.sumInputs())

plt.plot(N_array, totalSum)
plt.show()
