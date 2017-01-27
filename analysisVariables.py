## Features
measureDict = {0:'epsp_max', 1:'epsp_area', 2:'epsp_avg', 3:'epsp_time_to_peak', 4:'epsp_area_to_peak', 5:'epsp_min'} # This is always written as entityMeasured_measure
flagsList = [ "AP_flag", "noise_flag", "baseline_flag", "photodiode_flag"]
measureList = measureDict.keys()

## Plotting

plotTypeList = ['svg', 'png']
plotType = plotTypeList[1]
showPlots = 0
figsize=(20,20)
sliceImage = []
#sliceImage = inputDir + 'slice.tif'

## Experimental constants
F_sample = 20000 
threshold = 0.05
baseline= 100.
interest= 50.
samplingTime = 0.05

baselineWindowWidth = int(baseline/samplingTime)
interestWindowWidth = int(interest/samplingTime)

## Filtering
smootheningTime = 0.5 # In ms
filter={0:'',1:'ifft_bandpass',2:'bessel'}
filtering=filter[0]

## Controls
randomize_N_square_trials = False

if randomize_N_square_trials:
    suffix = '_randomized'
else:
    suffix = ''
