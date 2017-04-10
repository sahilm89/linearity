# Analysis related constants
randSeed = 202

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
interest= 100.
samplingTime = 0.05

baselineWindowWidth = int(baseline/samplingTime)
interestWindowWidth = int(interest/samplingTime)

## Filtering
smootheningTime = 0.5 # In ms
filter={0:'',1:'ifft_bandpass',2:'bessel'}
filtering=filter[2]

## Controls
randomize_N_square_trials = False

if randomize_N_square_trials:
    suffix = '_randomized'
else:
    suffix = ''
