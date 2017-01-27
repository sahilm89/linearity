#!/usr/bin/python
''' Read and use matlab files (.mat) to check if we can get the 5 squares stimulation information back.'''

import matplotlib.pyplot as plt
import numpy as np
import util
from statsmodels.graphics import tsaplots
import scipy
import scipy.signal as ss 
import rdp

def corr(x,y):
        x = x - x.mean()
        y = y - y.mean()
        autocorr = np.correlate(x, y, mode='full')
        autocorr = autocorr[x.size:]
        autocorr /= autocorr.max()
        return autocorr

#photo_150 = util.readMatlabFile('photoDiode/RS_PDtrace_150.mat')
#photo_200 = util.readMatlabFile('photoDiode/RS_PDtrace_200.mat')
#photo_250 = util.readMatlabFile('photoDiode/RS_PDtrace_250.mat')
#
#photodiode, null = util.parseDictKeys(photo_150)


#photo_200_100ms = util.readMatlabFile('/home/sahil/Documents/Codes/bgstimPlasticity/data/august/150901/200/100msblank.mat')
#photo_250_100ms = util.readMatlabFile('/home/sahil/Documents/Codes/bgstimPlasticity/data/august/150901/250/100msblank.mat')
#photo_200_200ms = util.readMatlabFile('/home/sahil/Documents/Codes/bgstimPlasticity/data/august/150901/200/200msblank.mat')
#photo_250_200ms = util.readMatlabFile('/home/sahil/Documents/Codes/bgstimPlasticity/data/august/150901/250/200msblank.mat')

#photodiode, null = util.parseDictKeys(photo_200_200ms)

photo_200_200ms = util.readMatlabFile('/home/sahil/Documents/Codes/bgstimPlasticity/data/august/150902/c1/20sec_PD.mat')

photodiode, null  = util.parseDictKeys(photo_200_200ms)

fullLength = []
#print photodiode

for key in photodiode.keys():
    photodiode[key] = photodiode[key].T
    fullLength.extend(photodiode[key][1])

#blank = fullLength[274960:277280]
#fullLength = photodiode[2].T[1]
lengthHalfLength =  len(fullLength)/50000
lengthFullLength =  len(fullLength)/2
#plt.plot(fullLength[:400100])
#plt.plot(fullLength[lengthHalfLength:lengthHalfLength+1000])
#plt.show()

#partialLengthVector1 = np.array(fullLength[:lengthHalfLength])
partialLengthVector2_1 = np.array(fullLength)

##fig, axes = plt.subplots(figsize=(8, 12))
##fig.tight_layout()
##axes.plot(fullLength[:lengthHalfLength])
##correlation = corr(partialLengthVector1, partialLengthVector2)
##correlation = ss.fftconvolve(partialLengthVector1, partialLengthVector2)
##tsaplots.plot_acf(correlation, axes)
##acorr(np.array(fullLength[:lengthHalfLength]),ax=ax)
##plt.show()
#
##freq = 2e4 # 20 kHz
##ts = 1/freq # Sampling time
##time = np.arange(0, ts*lengthFullLength, ts)
##
##k = np.arange(lengthFullLength)
##T = lengthFullLength/freq
##frq = k/T # two sides frequency range
##frq = frq[range(lengthFullLength/2)] # one side frequency range
##
##yf = np.fft.fft(fullLength)/lengthFullLength
##yf = yf[range(lengthFullLength/2)]
###yf = scipy.fftpack.fft(partialLengthVector1)
##xf = np.linspace(0.0, 1.0/(2.0*(0.05* lengthHalfLength ) ), lengthHalfLength/2)
#
##fig, ax = plt.subplots()
##ax.plot(xf[1:], 2.0/lengthHalfLength * np.abs(yf[0:lengthHalfLength/2])[1:])
##plt.show()
#
##fig, ax = plt.subplots(2, 1)
##ax[0].plot(time,fullLength)
##ax[0].set_xlabel('Time')
##ax[0].set_ylabel('Amplitude')
##ax[1].plot(frq,abs(yf),'r') # plotting the spectrum
##ax[1].set_xlabel('Freq (Hz)')
##ax[1].set_ylabel('|Y(freq)|')
##plt.show()
#
##print frq
##frq_filtered = frq[5:12]
##filtered_yf = yf[5:12] 
##filtered_yt = np.fft.ifft(filtered_yf)
##t = np.arange(7)
##print len(t), len(filtered_yt)
##plt.plot(t, filtered_yt.real, 'b-', t, filtered_yt.imag, 'r--')
##plt.show()
#
#Spectrum, Filtered_spectrum, Filtered_signal, Low_point, High_point = util.bandpass_ifft(np.array(fullLength), 0.04,0.1, 20000)
#
#fig1 = plt.figure()
#plt.plot(time, abs(Filtered_signal))
#plt.show()

t = np.arange(0,len(partialLengthVector2_1)/20,0.05)
Spectrum, Filtered_spectrum, Filtered_signal_1, Low_point, High_point = util.bandpass_ifft(np.array(partialLengthVector2_1), 0, 500, 20000, nfactor=80 )
#Spectrum, Filtered_spectrum, Filtered_signal_1, Low_point, High_point = util.bandpass_ifft(np.array(partialLengthVector2_1), 0, 300, 20000)

#photodiode, null = util.parseDictKeys(photo_200_200ms)
###################2 ######################
#photo_200_200ms = util.readMatlabFile('/home/sahil/Documents/Codes/bgstimPlasticity/data/august/150828/c1/RS_135.mat')
#photodiode, null = util.parseDictKeys(photo_250_200ms)
#
#fullLength = []
#print photodiode
#
#for key in photodiode.keys():
#    photodiode[key] = photodiode[key].T
#    print len(photodiode[key][1]) 
#    fullLength.extend(photodiode[key][1])
#
##blank = fullLength[274960:277280]
##fullLength = photodiode[2].T[1]
#lengthHalfLength =  len(fullLength)/50000
#lengthFullLength =  len(fullLength)/2
##plt.plot(fullLength[:400100])
##plt.plot(fullLength[lengthHalfLength:lengthHalfLength+1000])
##plt.show()
#
##partialLengthVector1 = np.array(fullLength[:lengthHalfLength])
#partialLengthVector2_2 = np.array(fullLength)
#t = np.arange(0,len(partialLengthVector2_2)/20,0.05)
#Spectrum, Filtered_spectrum, Filtered_signal_2, Low_point, High_point = util.bandpass_ifft(np.array(partialLengthVector2_2), 0, 150, 20000)


#photodiode, null = util.parseDictKeys(photo_200_200ms)

#photo_200_200ms = util.readMatlabFile('/home/sahil/Documents/Codes/bgstimPlasticity/data/august/150828/c1/RS_135.mat')
#photodiode, null = util.parseDictKeys(photo_200_100ms)

#fullLength = []
#print photodiode
#
#for key in photodiode.keys():
#    photodiode[key] = photodiode[key].T
#    print len(photodiode[key][1]) 
#    fullLength.extend(photodiode[key][1])
#
##blank = fullLength[274960:277280]
##fullLength = photodiode[2].T[1]
#lengthHalfLength =  len(fullLength)/50000
#lengthFullLength =  len(fullLength)/2
##plt.plot(fullLength[:400100])
##plt.plot(fullLength[lengthHalfLength:lengthHalfLength+1000])
##plt.show()
#
##partialLengthVector1 = np.array(fullLength[:lengthHalfLength])
#partialLengthVector2_3 = np.array(fullLength)
#t = np.arange(0,len(partialLengthVector2_3)/20,0.05)
#Spectrum, Filtered_spectrum, Filtered_signal_3, Low_point, High_point = util.bandpass_ifft(np.array(partialLengthVector2_3), 0, 150, 20000)


indices = scipy.signal.find_peaks_cwt(Filtered_signal_1,np.arange(1,5),noise_perc=0)
print len(indices)
t2 = np.arange(0,len(Filtered_signal_1))
#plt.figure()
#plt.plot(partialLengthVector2_1, 'b')
plt.plot(t2, Filtered_signal_1, 'g')
plt.plot(t2[indices],Filtered_signal_1[indices],'rD')
##plt.subplot(312)
##plt.plot(t, partialLengthVector2_2, 'b')
##plt.plot(t, Filtered_signal_2, 'g')
##plt.subplot(313)
##plt.plot(t, partialLengthVector2_3, 'b')
##plt.plot(t, Filtered_signal_3, 'g')
#
plt.show()

#fullFiltered_signal = Filtered_signal

######### Doing this for the blank #################

#t = np.arange(0,len(blank)/20,0.05)
#Spectrum, Filtered_spectrum, Filtered_signal, Low_point, High_point = util.bandpass_ifft(np.array(blank), 0, 150, 20000)
#plt.figure()
#
##plt.subplot(211)
#plt.plot(t, blank, 'b')
##plt.subplot(212)
#plt.plot(t, Filtered_signal, 'g')
#plt.show()
######################################################
#
##fig, axes = plt.subplots(figsize=(8, 12))
##fig.tight_layout()
##axes.plot(fullLength[:lengthHalfLength])
##correlation = corr(Filtered_signal,partialLengthVector2)
#correlation = ss.fftconvolve(fullFiltered_signal,Filtered_signal)
#tsaplots.plot_acf(correlation)
#plt.show()

