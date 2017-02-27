import numpy as np
import scipy.stats as ss
import itertools as it
import pickle   # Used to save complete Neuron objects
from collections import OrderedDict as ordered
import warnings
import os

class Neuron:
    ''' This is Experiment class for the photostimulation experiment '''

    def __init__(self, index, date):
        self.index = index
        self.date = date
        self.experiment = {}
        self.features = {0: 'epsp_max', 1: 'epsp_area', 2: 'epsp_avg',
                            3: 'epsp_time_to_peak', 4: 'epsp_area_to_peak',
                            5: 'epsp_min', 6: 'epsp_onset'}
        self.flagsList = ["AP_flag", "noise_flag", "baseline_flag",
                          "photodiode_flag"]

    def analyzeExperiment(self, type, squares, voltage, photodiode, coords,
                          marginOfBaseLine, marginOfInterest,
                          F_sample, smootheningTime):
        if type not in self.experiment:
            self.experiment[type] = {squares: Experiment(self, type, squares,
                                     voltage, photodiode, coords,
                                     marginOfBaseLine, marginOfInterest,
                                     F_sample, smootheningTime)}
        else:
            self.experiment[type].update({squares: Experiment(self, type,
                                          squares, voltage, photodiode,
                                          coords, marginOfBaseLine,
                                          marginOfInterest, F_sample,
                                          smootheningTime)})
        # self.experiment[type][squares]._transformTrials()
        self.experiment[type][squares]._groupTrialsByCoords()  # Coord grouping
        # self.experiment[type][squares]._transformCoords()
        if not squares == 1:
            self.experiment[type][squares]._findRegressionCoefficients()

    def save(self, filename):
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

class Experiment:
    ''' Change this to key: [coords, photodiode, value]'''
    def __init__(self, neuron, type, numSquares, voltage, photodiode, coords,
                 marginOfBaseline, marginOfInterest,
                 F_sample, smootheningTime):
        self.neuron = neuron
        self.type = type
        self.numSquares = numSquares
        self.F_sample = F_sample
        self.samplingTime = 1./self.F_sample
        self.smootheningTime = smootheningTime
        self.marginOfBaseline = marginOfBaseline
        self.marginOfInterest = marginOfInterest
        self.coords = self._returnCoordinates(coords, voltage)
        self.coordwise = {}
        self.regression_coefficients = {}
        self.trial = {index: Trial(self, index, self.coords[index],
                      photodiode[index].T[1], voltage[index].T[1])
                      for index in voltage}

    def _returnCoordinates(self, coords, voltage):
        ''' Setup the coordinates for the trial '''
        keyrange = sorted(voltage.keys())
        assert keyrange == range(1, keyrange[-1]+1)  # Check missing trials
        xy = it.cycle(coords)  # Indefinite cycling through the coordinates.
        coords_for_square = ordered({})

        for key in keyrange:
            tempCoord = []
            for i in range(self.numSquares):
                tempCoord.append(xy.next())

            coords_for_square[key] = frozenset(tempCoord)
            if len(coords_for_square[key]) != self.numSquares:
                warnings.warn("Duplicate coordinate {} in combination {}!"
                              .format(tempCoord, coords_for_square[key]))
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
                    self.trial[index].linearly_transformed_feature[featureIndex] =\
                            self.trial[index]._linearTransform(
                                    self.trial[index].feature[featureIndex],
                                    minValue, maxValue)

    def _transformCoords(self):
        ''' Setup transformations on coordinates'''
        for featureIndex in self.neuron.features.keys():
            coordValues_average = []
            coordValues_expected = []

            # Average Features for all coords
            for coord in self.coordwise.keys():
                coordValues_average.append(self.coordwise[coord].average_feature[featureIndex])
            min_average, max_average = min(coordValues_average), max(coordValues_average)
            for coord in self.coordwise.keys():
                self.coordwise[coord].linearly_transformed_average_feature[featureIndex] =\
                        self.coordwise[coord]._linearTransform(
                                self.coordwise[coord].average_feature[featureIndex],
                                min_average, max_average)

            # Expected Features for >1 numSquares
            if not self.numSquares == 1:
                for coord in self.coordwise.keys():
                        coordValues_expected.append(
                                self.coordwise[coord].expected_feature[featureIndex])
                min_expected, max_expected = min(coordValues_expected), max(coordValues_expected)
                for coord in self.coordwise.keys():
                        self.coordwise[coord].linearly_transformed_expected_feature[featureIndex] =\
                                self.coordwise[coord]._linearTransform(
                                        self.coordwise[coord].expected_feature[featureIndex],
                                        min_expected, max_expected)

    def _groupTrialsByCoords(self):
        coordwise = {}
        for trial in self.trial:
            if self.trial[trial].coord not in coordwise:
                coordwise.update({self.trial[trial].coord: [self.trial[trial]]})
            else:
                coordwise[self.trial[trial].coord].append(self.trial[trial])
        self.coordwise = {coord: Coordinate(coord, coordwise[coord], self)
                          for coord in coordwise}

    def _findRegressionCoefficients(self):
        for feature in self.neuron.features:
            expected = []
            observed = []
            for coord in self.coordwise.keys():
                if feature in self.coordwise[coord].feature:
                    expected.append(self.coordwise[coord].expected_feature[feature])
                    observed.append(self.coordwise[coord].average_feature[feature])
            self.regression_coefficients.update(
                    {feature: {key: value for key, value in
                               zip(['slope', 'intercept', 'r_val', 'p_val', 'stderr'],
                                   ss.linregress(expected, observed))}})

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
        # self.photodiode = photodiode
        # self.voltage = voltage
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
        print self.index, self.coord

        # All features here, move some features out of this for APs
        if not (self.AP_flag or self.baseline_flag or self.photodiode_flag):
            for featureIndex in xrange(len(self.neuron.features)):
                self.feature.update({featureIndex: self.extractFeature(featureIndex)})
        # print self.index, self.feature[0]

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

    # Features
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
        '''Finds the area under the curve of the vector in the given window.
           This will subtract negative area from the total area.'''
        auc = np.trapz(self.interestWindow, dx=self.samplingTime)  # in V.s
        return auc

    def _areaUnderTheCurveToPeak(self, samplingFreq):
        '''Finds the area under the curve of the vector in the given window'''
        maxIndex = np.argmax(self.interestWindow)
        windowToPeak = self.interestWindow[:maxIndex+1]
        auctp = np.trapz(windowToPeak, dx=self.samplingTime)  # in V.s
        return auctp

    def _findOnsetTime(self, samplingFreq, steps=50, pValTolerance=0.01):
        ''' Find the onset of the curve using a 2 sample KS test '''
        # print "_findOnset doesn't work yet!"
        window_size = len(self.interestWindow_raw)
        step_size = window_size/steps
        index_right = step_size
        for index_left in xrange(0, window_size, step_size):
            stat, pVal = ss.ks_2samp(self.baselineWindow, self.interestWindow_raw[index_left:index_right])
            index_right += step_size
            if pVal < pValTolerance:
                # print index_left, pVal, stat#, self.interestWindow_raw[index_left:index_right]
                break
        return float(index_left)/samplingFreq

    # Flags
    def _flagActionPotentials(self, AP_threshold=3e-2):
        ''' This function flags if there is an AP trialwise and returns a dict of bools '''
        if np.max(self.interestWindow) > AP_threshold:
            print "Action Potential in trial {}".format(self.index)
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

    def _flagNoise(self, pValTolerance=0.01):
        ''' This function asseses if the distributions of the baseline and interest are different or not '''
        m, pVal = ss.ks_2samp(self.baselineWindow, self.interestWindow)
        if pVal < pValTolerance:
            return 0
        else:
            print "No response measured in trial {}".format(self.index)
            return 1  # Flagged as noisy

    # Transformations
    def _linearTransform(self, value, minValue, maxValue):
        return (value - minValue)/(maxValue - minValue)

    def _normalizeToBaseline(self, interestWindow, baselineWindow):
        '''normalizes the vector to an average baseline'''
        baseline = np.average(baselineWindow)
        interestWindow_new = interestWindow - baseline  # Subtracting baseline from whole array
        return interestWindow_new, baseline

    def _smoothen(self, smootheningTime, samplingFreq):
        '''normalizes the vector to an average baseline'''
        smootheningWindow = smootheningTime*1e-3*samplingFreq
        window = np.ones(int(smootheningWindow)) / float(smootheningWindow)
        self.interestWindow = np.convolve(self.interestWindow, window, 'same')  # Convolving with a rectangle


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
        self.feature = self._setFeatures()

        if not self.numSquares == 1:
            self._findExpected()

    def _flagCoordinate(self):
        ''' Flags the coordinate for all flags '''

        # Initializing flags for this coordinate
        for flag in self.neuron.flagsList:
            setattr(self, flag, 1)

        # Setting actual flags for this coordinate
        for flag in self.neuron.flagsList:
            for trial in self.trials:
                setattr(self, flag, getattr(self, flag)*trial.flags[flag])
            self.flags.update({flag: getattr(self, flag)})

    def _findAveraged(self):
        ''' Finds the average feature for the coord '''
        for feature in self.neuron.features:
            averageFeature = np.average([trial.feature[feature]
                                         for trial in self.trials
                                         if feature in trial.feature])
            if not np.isnan(averageFeature):
                self.average_feature.update({feature: averageFeature})

    def _setFeatures(self):
        ''' Sets valid features for the coordinate based on flags '''
        return self.average_feature.keys()

    def _findExpected(self):
        ''' Finds the expected feature from one squares data '''
        if 1 in self.neuron.experiment[self.type].keys():
            for feature in self.feature:
                sum_of_features = 0.
                for coord in self.coords:
                    oneSquareCoordwise = self.neuron.experiment[self.type][1].coordwise
                    if frozenset([coord]) in oneSquareCoordwise.keys():
                        if feature in oneSquareCoordwise[frozenset([coord])].average_feature:
                            sum_of_features += oneSquareCoordwise[frozenset([coord])].average_feature[feature]
                        else:
                            sum_of_features = None
                            break
                if sum_of_features is not None:
                    self.expected_feature.update({feature: sum_of_features})

    def _linearTransform(self, value, minValue, maxValue):
        return (value - minValue)/(maxValue - minValue)
