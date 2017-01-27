'''General Analysis and plotting file for voltage traces from Aanchal's
Random Simulation project '''

from util import readMatlabFile, parseDictKeys, find_BaseLine_and_WindowOfInterest_Margins, convertDictToList, plotTrace, bandpass_ifft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA


file = '/home/sahil/Documents/Codes/bgstimPlasticity/data/august/150827/c4/CPP.mat'

samplingTime = 0.05 # ms

threshold = 0.05  # 50 mV 
smootheningTime = 0.25 # ms
baseline= 100. # ms
interest= 50. # ms

smootheningWindow = int(smootheningTime/samplingTime )
baselineWindowWidth = int(baseline/samplingTime)
interestWindowWidth = int(interest/samplingTime)


dict = readMatlabFile(file)
volt, photodiode = parseDictKeys(dict)
trace = convertDictToList(volt)

baseLineWindow, interestWindow = find_BaseLine_and_WindowOfInterest_Margins(photodiode, threshold, baselineWindowWidth, interestWindowWidth)
###############################################################################
# Compute ICA
voltageTrace = [traceIndex[1][:baseLineWindow[1]] for traceIndex in trace]
transposedTraces = map(list,zip(*voltageTrace))
'''
X = np.array(transposedTraces)

numComponents = 5 
ica = FastICA(n_components=numComponents)
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
#assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

pca = PCA(n_components=numComponents)
H = pca.fit_transform(X)
###############################################################################
# Plot results

plt.figure()

models = [X, S_, H]
names =  ['original data', 'ICA recovered signals', 'pca']
colors = ['red', 'steelblue', 'green']


for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(3, 1, ii)
        plt.title(name)
        for sig in model.T:
            plt.plot(sig, label= name,alpha=0.5)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()

t = np.arange(len(voltageTrace[0]))
sp = np.fft.fftn(voltageTrace)
freq = np.fft.fftfreq(t.shape[-1])

for k in sp:
    plt.plot(freq, k.real, freq, k.imag)
plt.show()

'''
print photodiode
voltageTrace = [traceIndex[1][:baseLineWindow[1]] for traceIndex in trace]
t = np.arange(len(voltageTrace[0]))
Spectrum, Filtered_spectrum, Filtered_signal, Low_point, High_point = bandpass_ifft(np.array(voltageTrace[0]), 0, 120, 20000)

plt.figure()

plt.plot(t, voltageTrace[0], 'b')
plt.plot(t, Filtered_signal, 'r--')
plt.show()
