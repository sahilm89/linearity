import numpy as np
import scipy
import scipy.stats as ss
from scipy.optimize import curve_fit
import lmfit
import itertools as it
import pickle # Can be used to save entire objects
from collections import OrderedDict as ordered 
import matplotlib.pyplot as plt

class Neuron:
    ''' This is Experiment class for the photostimulation experiment '''
    def __init__(self, index, date):
        self.index = index
        self.date = date
        self.experiment = {}
        self.features = {0:'epsp_max', 1:'epsp_area', 2:'epsp_avg', 3:'epsp_time_to_peak', 4:'epsp_area_to_peak', 5:'epsp_min', 6:'epsp_onset'} # This is always written as entityfeatured_measure
        self.flagsList = [ "AP_flag", "noise_flag", "baseline_flag", "photodiode_flag"]

    def analyzeExperiment(self, type, squares, voltage, photodiode, coords, marginOfBaseLine, marginOfInterest, F_sample, smootheningTime):
        if not type in self.experiment:
            self.experiment[type] = {squares: Experiment(self, type, squares, voltage, photodiode, coords, marginOfBaseLine, marginOfInterest,F_sample, smootheningTime )}
        else:
            self.experiment[type].update({squares: Experiment(self, type, squares, voltage, photodiode, coords, marginOfBaseLine, marginOfInterest, F_sample, smootheningTime )})
        #self.experiment[type][squares]._transformTrials()
        self.experiment[type][squares]._groupTrialsByCoords() #Creating coord based grouping
        #self.experiment[type][squares]._transformCoords()
        if not squares == 1: 
            self.experiment[type][squares]._findRegressionCoefficients() #Find regression coefficients

    def save(self,filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

class Experiment:
    ''' Change this to key: [coords, photodiode, value]''' 
    def __init__(self, neuron, type, numSquares, voltage, photodiode, coords, marginOfBaseline, marginOfInterest, F_sample, smootheningTime):
        self.neuron = neuron
        if type:
            self.type = type 
        else:
            self.type = 'Control'
        self.numSquares = numSquares 
        self.F_sample = F_sample
        self.samplingTime = 1./self.F_sample
        self.smootheningTime = smootheningTime
        self.marginOfBaseline = marginOfBaseline
        self.marginOfInterest = marginOfInterest
        self.coords = self._returnCoordinates(coords, voltage)
        self.coordwise = {}
        self.regression_coefficients = {}
        self.trial = {index : Trial(self, index, self.coords[index], photodiode[index].T[1], voltage[index].T[1]) for index in voltage}

    def _returnCoordinates(self, coords, voltage):
        ''' Setup the coordinates for the trial '''
        keyrange = sorted(voltage.keys())
        assert keyrange == range(1, keyrange[-1]+1) ## Making sure no trials are missing
        xy = it.cycle(coords) # This will keep doing an indefinite cycling through the coordinates.
        coords_for_square = ordered({})

        for key in keyrange:
            tempCoord = []
            for i in range(self.numSquares):
                tempCoord.append(xy.next())

            coords_for_square[key] = frozenset(tempCoord)
            #assert len(coords_for_square[key]) == self.numSquares, "Same coordinate multiple times in the combination!" # Making sure no coordinates repeat in the same square
            if len(coords_for_square[key]) != self.numSquares:
                print tempCoord, coords_for_square[key], "Same coordinate multiple times in the combination!" # Making sure no coordinates repeat in the same square
        return coords_for_square

    def _transformTrials(self):
        ''' Setup transformations on Trials'''
        for featureIndex in self.neuron.features.keys():
            trialValues = []
            for index in self.trial.keys():
                if featureIndex in self.trial[index].feature:
                    trialValues.append(self.trial[index].feature[featureIndex])
            minValue, maxValue = min(trialValues), max(trialValues)
            for index in self.trial.keys():
                if featureIndex in self.trial[index].feature:
                    self.trial[index].linearly_transformed_feature[featureIndex] = self.trial[index]._linearTransform(self.trial[index].feature[featureIndex], minValue, maxValue)

    def _transformCoords(self):
        ''' Setup transformations on coordinates'''
        for featureIndex in self.neuron.features.keys():
            coordValues_average = []
            coordValues_expected = []

            ########## Average Features for all coords ###
            for coord in self.coordwise.keys():
                coordValues_average.append(self.coordwise[coord].average_feature[featureIndex])
            min_average, max_average = min(coordValues_average), max(coordValues_average)
            for coord in self.coordwise.keys():
                self.coordwise[coord].linearly_transformed_average_feature[featureIndex] = self.coordwise[coord]._linearTransform(self.coordwise[coord].average_feature[featureIndex], min_average, max_average)

            ####### Expected Features for >1 numSquares
            if not self.numSquares == 1: 
                for coord in self.coordwise.keys():
                        coordValues_expected.append(self.coordwise[coord].expected_feature[featureIndex])
                min_expected, max_expected = min(coordValues_expected), max(coordValues_expected)
                for coord in self.coordwise.keys():
                        self.coordwise[coord].linearly_transformed_expected_feature[featureIndex] = self.coordwise[coord]._linearTransform(self.coordwise[coord].expected_feature[featureIndex], min_expected, max_expected)

    def _groupTrialsByCoords(self):
        coordwise = {}
        for trial in self.trial:
            if self.trial[trial].coord not in self.coordwise:
                coordwise.update({self.trial[trial].coord: [self.trial[trial]]})
            else:
                coordwise[self.trial[trial].coord].append(self.trial[trial])
        self.coordwise = {coord: Coordinate(coord, coordwise[coord], self) for coord in coordwise}

    def _findRegressionCoefficients(self):
        for feature in self.neuron.features: 
            expected = []
            observed = []
            for coord in self.coordwise.keys(): 
                if feature in self.coordwise[coord].feature:
                    expected.append(self.coordwise[coord].expected_feature[feature])
                    observed.append(self.coordwise[coord].average_feature[feature])
            self.regression_coefficients.update({feature : {key:value for key, value in zip(['slope', 'intercept', 'r_val', 'p_val', 'stderr'] , ss.linregress(expected, observed))}})

#    def _findAverageFeatures(self):
#        ''' Finds the expected feature from one squares data '''
#        for feature in self.neuron.features:
#            self.coordwise_feature[feature] = {}
#            for trial in self.trial:
#                if self.trial[trial].coord not in self.coordwise_feature[feature]:
#                    self.coordwise_feature[feature].update({self.trial[trial].coord: [self.trial[trial].feature[feature]]})
#                else:
#                    self.coordwise_feature[feature][trial.coord].append(self.trial[trial].feature[feature])
#            for key in self.coordwise_feature[feature]:
#                self.average_feature[feature].update({key: np.average(self.average_feature[feature][key])})

class Trial:
    def __init__(self, experiment, index, coord, photodiode, voltage):
        self.index = index
        self.coord = coord
        #self.photodiode = photodiode
        #self.voltage = voltage
        self.experiment = experiment 
        self.neuron = self.experiment.neuron
        self.F_sample = self.experiment.F_sample
        self.samplingTime = self.experiment.samplingTime
        self.smootheningTime = self.experiment.smootheningTime
        self.feature = {}
        self.linearly_transformed_feature = {} 
        self.flags = {}

        self.interestWindow_raw = voltage[self.experiment.marginOfInterest[0]:self.experiment.marginOfInterest[1]]
        self.baselineWindow = voltage[self.experiment.marginOfBaseline[0]:self.experiment.marginOfBaseline[1]]
        self.interestWindow, self.baseline = self._normalizeToBaseline(self.interestWindow_raw, self.baselineWindow)
        self.setupFlags()
        self._smoothen(self.smootheningTime, self.F_sample)
        if self.experiment.type == "GABAzine":
            normalized_interestWindow = self.interestWindow/np.mean(self.interestWindow)
            #normalized_interestWindow = self.interestWindow + 0.
            time = np.arange(len(self.interestWindow))*self.samplingTime
            self.fit_using_lmfit (time, normalized_interestWindow,"xyz")
            #popt, pcov = self.fitFunctionToPSP(time, normalized_interestWindow, "double_exponential")
            #print popt, pcov
            #plt.plot(time, normalized_interestWindow, alpha=0.2)
            #psp_time = time[time > popt[0]]
            #pre_psp_time = time[time <= popt[0]]
            #plt.plot(psp_time, self._doubleExponentialFunction(psp_time,*popt))
            #plt.plot(pre_psp_time, np.zeros(len(pre_psp_time)))
            #plt.show()

        # All features here, move some features out of this for APs
        if not (self.AP_flag or self.baseline_flag or self.photodiode_flag):
            for featureIndex in xrange(len(self.neuron.features)):
                self.feature.update({featureIndex:self.extractFeature(featureIndex)})
        #print self.index, self.feature[0]
    def setupFlags(self):
        ''' Setup all flags for trial '''
        for flag in self.neuron.flagsList:
            if flag == "AP_flag":
                setattr(self, flag, self._flagActionPotentials())
            elif flag == "noise_flag":
                setattr(self, flag, self._flagNoise())
            elif flag == "baseline_flag":
                setattr(self, flag, self._flagBaseLineInstability())
            elif flag == "photodiode_flag":
                setattr(self, flag, self._flagPhotodiodeInstability())
            self.flags.update({flag: getattr(self, flag)})

    def extractFeature(self, feature):

        if feature == 0:
            return self._findMaximum()
        elif feature == 1:
            return self._areaUnderTheCurve(self.F_sample)
        elif feature == 2:
            return self._findMean()
        elif feature == 3:
            return self._findTimeToPeak(self.F_sample)
        elif feature == 4:
            return self._areaUnderTheCurveToPeak(self.F_sample)
        elif feature == 5:
            return self._findMinimum()
        elif feature == 6:
            return self._findOnsetTime(self.F_sample)


    ################## Features ########################
    def _findMaximum(self):
        '''Finds the maximum of the vector in a given interest'''
        return np.max(self.interestWindow)
    
    def _findMinimum(self):
        '''Finds the maximum of the vector in a given interest'''
        return np.min(self.interestWindow)
    
    def _findTimeToPeak(self, samplingFreq):
        '''Finds the time to maximum of the vector in a given interest'''
        maxIndex = np.argmax(self.interestWindow)
        timeToPeak = (maxIndex)*self.samplingTime
        return timeToPeak 
    
    def _findMean(self):
        '''Finds the mean of the vector in a given interest'''
        return np.average(self.interestWindow)
    
    def _areaUnderTheCurve(self, samplingFreq):
        '''Finds the area under the curve of the vector in the given window. This will subtract negative area from the total area.'''
        auc = np.trapz(self.interestWindow,dx=self.samplingTime) # in V.s
        return auc

    def _areaUnderTheCurveToPeak(self, samplingFreq):
        '''Finds the area under the curve of the vector in the given window'''
        maxIndex = np.argmax(self.interestWindow)
        windowToPeak = self.interestWindow[:maxIndex+1] 
        auctp = np.trapz(windowToPeak,dx=self.samplingTime) # in V.s
        return auctp

    def _findOnsetTime(self, samplingFreq, steps= 50, pValTolerance = 0.01):
        ''' Find the onset of the curve using a 2 sample KS test '''
        print "_findOnset doesn't work yet!"
        window_size = len(self.interestWindow_raw)
        step_size = window_size/steps
        index_right = step_size
        for index_left in xrange(0, window_size, step_size):
            stat, pVal = ss.ks_2samp(self.baselineWindow, self.interestWindow_raw[index_left:index_right])
            index_right += step_size
            if pVal<pValTolerance:
                print index_left, pVal, stat#, self.interestWindow_raw[index_left:index_right]
                break
        return float(index_left)/samplingFreq


    ###################### Flags ############################
    def _flagActionPotentials(self, AP_threshold=3e-2):
        ''' This function flags if there is an AP trialwise and returns a dict of bools '''
        if np.max(self.interestWindow) > AP_threshold:
            return 1 
        else:
            return 0
    
    def _flagBaseLineInstability(self):
        # Takes voltageTrace, baseline
        ''' This function flags if the baseline is too variable for measuring against '''
        return 0
    
    def _flagPhotodiodeInstability(self):
        # Takes photodiode, margin
        ''' This function flags if the photodiode trace is too noisy '''
        return 0
    
    def _flagNoise(self, pValTolerance = 0.05):
        ''' This function asseses if the distributions of the baseline and interest are different or not '''
        m, pVal = ss.ks_2samp(self.baselineWindow, self.interestWindow)
        if pVal < pValTolerance:
            return 0
        else:
            return 1 # Flagged as noisy

    ## Transformations
    def _linearTransform(self, value, minValue, maxValue):
        return (value - minValue)/(maxValue - minValue)

    def _normalizeToBaseline(self, interestWindow, baselineWindow):
        '''normalizes the vector to an average baseline'''
        baseline = np.average(baselineWindow)
        interestWindow_new = interestWindow - baseline # Subtracting baseline from whole array
        return interestWindow_new, baseline

    def _smoothen(self, smootheningTime, samplingFreq):
        '''normalizes the vector to an average baseline'''
        smootheningWindow = smootheningTime*1e-3*samplingFreq
        window = np.ones(int(smootheningWindow)) / float(smootheningWindow)
        self.interestWindow = np.convolve(self.interestWindow, window, 'same')  # Subtracting baseline from whole array

    ## Fits
    print "Gabazine Fits are okay, but cleanup is required at this point"
    def _doubleExponentialFunction(self, t, t_0, tOn, tOff, g_max):
        ''' Returns the shape of an EPSP as a double exponential function '''
        tPeak = t_0 + float(((tOff * tOn)/(tOff-tOn)) * np.log(tOff/tOn))
        A = 1./(np.exp(-(tPeak-t_0)/tOff) - np.exp(-(tPeak-t_0)/tOn))
        #g = g_max * A * (np.exp(-(t-t_0)/tOff) - np.exp(-(t-t_0)/tOn))
        g = [ g_max * A * (np.exp(-(t_point-t_0)/tOff) - np.exp(-(t_point-t_0)/tOn)) if  t_point >= t_0 else 0.  for t_point in t]
        return g
    
    def fitFunctionToPSP(self, time, vector, function):
        if function == "double_exponential":
            popt,pcov = curve_fit(self._doubleExponentialFunction,time,vector, bounds=(0, [max(time), max(time), max(time), max(vector)]), p0 = (max(time)/2., max(time)/4., max(time)/3., max(vector)))
            #popt,pcov = curve_fit(self._doubleExponentialFunction,time,vector, p0 = (max(time)/2., max(time)/4., max(time)/3., max(vector)))
        else:
            print "No other function yet"
        return popt, pcov

    def fit_using_lmfit(self, time, vector, function):
        ''' Fits using lmfit '''
        def _doubleExponentialFunction(t, t_0, tOn, tOff, g_max):
            ''' Returns the shape of an EPSP as a double exponential function '''
            tPeak = t_0 + float(((tOff * tOn)/(tOff-tOn)) * np.log(tOff/tOn))
            A = 1./(np.exp(-(tPeak-t_0)/tOff) - np.exp(-(tPeak-t_0)/tOn))
            g = [ g_max * A * (np.exp(-(t_point-t_0)/tOff) - np.exp(-(t_point-t_0)/tOn)) if  t_point >= t_0 else 0.  for t_point in t]
        return g

        model = lmfit.Model(_doubleExponentialFunction)
        model.set_param_hint('t_0', value =max(time)/10., min=0., max = max(time))
        model.set_param_hint('tOn', value =max(time)/5.1 , min = 0., max = max(time))
        model.set_param_hint('tOff', value =max(time)/5. , min = 0., max = max(time))
        model.set_param_hint('g_max', value = max(vector)/1.1, min = 0., max = max(vector))
        pars = model.make_params()

        result = model.fit(vector, pars, t=time )
        print(result.fit_report())
        ax = plt.subplot(111)
        ax.plot(time, vector, alpha=0.2)
        ax.plot(time, result.best_fit, '-')
        plt.show()

class Coordinate:
    def __init__(self, coords, trials, experiment):
        self.coords = coords
        self.trials = trials
        self.experiment = experiment
        self.numSquares = self.experiment.numSquares
        self.neuron = self.experiment.neuron
        self.type = self.experiment.type
        self.average_feature = {}
        self.expected_feature = {}
        self.linearly_transformed_expected_feature = {} 
        self.linearly_transformed_average_feature = {} 
        self.flags = {}

        self._findAveraged()
        self.feature = self. _setFeatures()

        if not self.numSquares == 1:
            self._findExpected()

    def _flagCoordinate(self):
        ''' Flags the coordinate for all flags '''

        # Initializing flags for this coordinate
        for flag in flagList:
            setattr(self, flag, 1)

        # Setting actual flags for this coordinate
        for flag in self.neuron.flagsList:
            for trial in self.trials:
                setattr(self, flag, getattr(self, flag)*trial.flags[flag])
            self.flags.update({flag: getattr(self, flag)})

    def _findAveraged(self):
        ''' Finds the average feature for the coord '''
        for feature in self.neuron.features:
            averageFeature = np.average([trial.feature[feature] for trial in self.trials if feature in trial.feature])
            if not np.isnan(averageFeature):
                self.average_feature.update({feature : averageFeature})

    def _setFeatures(self):
        ''' Sets valid features for the coordinate based on flags '''
        return self.average_feature.keys()

    def _findExpected(self):
        ''' Finds the expected feature from one squares data '''
        for feature in self.feature:
            self.expected_feature.update({feature: np.sum([self.neuron.experiment[self.type][1].coordwise[frozenset([coord])].average_feature[feature] for coord in self.coords])})

    def _linearTransform(self, value, minValue, maxValue):
        return (value - minValue)/(maxValue - minValue)
