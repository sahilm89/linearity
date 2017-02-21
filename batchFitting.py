import glob
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.stats as ss
plt.style.use('seaborn-white')
import statsmodels.api as sm

filelist = glob.glob('/media/sahil/NCBS_Shares_BGStim/patch_data/*/c?/plots/*.pkl')
#f, (ax1, ax2) = plt.subplots(1, 2)
result2_rsquared_adj = []
result1_rsquared_adj = []
var_expected = []
tolerance = 5e-4
for i, file in enumerate(filelist):
    print file
    try:
        control_observed = {}
        gabazine_observed ={}
        control_expected = {}
        gabazine_expected ={}
        feature = 2

        with open(file, 'rb') as input:
            neuron = pickle.load(input)
            for type in neuron.experiment:
                for numSquares in neuron.experiment[type].keys(): 
                    if not numSquares == 1:
                        nSquareData = neuron.experiment[type][numSquares]
                        if type == "Control":
                            coords_C = nSquareData.coordwise
                            for coord in coords_C: 
                                if feature in coords_C[coord].feature:
                                    control_observed.update({coord: []})
                                    control_expected.update({coord: []})
                                    for trial in coords_C[coord].trials:
                                        control_observed[coord].append(trial.feature[feature])
                                        control_expected[coord].append(coords_C[coord].expected_feature[feature])
                        elif type == "GABAzine":
                            coords_I = nSquareData.coordwise
                            for coord in coords_I: 
                                if feature in coords_I[coord].feature:
                                    gabazine_observed.update({coord: []})
                                    gabazine_expected.update({coord: []})
                                    for trial in coords_I[coord].trials:
                                        gabazine_observed[coord].append(trial.feature[feature])
                                        gabazine_expected[coord].append(coords_I[coord].expected_feature[feature])
            print "Read {} into variables".format(file)
    except:
        print "Some problem with this file. Check {}! ".format(file)
        continue

    list_control_observed   = []  
    list_gabazine_observed  = []
    list_control_expected   = []
    list_gabazine_expected  = []

    if len(gabazine_observed):
        for key in gabazine_observed.keys():
            for element1, element2 in zip(gabazine_observed[key], gabazine_expected[key] ):
                if not (element1 <0 or np.isclose(element1, 0, atol=tolerance) or element2<0 or np.isclose(element2, 0, atol=tolerance)):
                    list_gabazine_observed.append(element1)
                    list_gabazine_expected.append(element2)

    print len(control_observed)
    if len(control_observed):
        for key in control_observed.keys():
            for element1, element2 in zip(control_observed[key], control_expected[key] ):
                if not (element1 <0 or np.isclose(element1, 0, atol=tolerance) or element2<0 or np.isclose(element2, 0, atol=tolerance)):
                    list_control_observed.append(element1)
                    list_control_expected.append(element2)
            
        if len(list_control_expected)>30 and len(list_control_observed)>30:
            X = np.array(list_control_expected)
            y = np.array(list_control_observed)
            X_log = np.log10(list_control_expected)

            const_X = sm.add_constant(X)
            const_X_log = sm.add_constant(X_log)

            #const_X = X
            #const_X_log = X_log

            linearModel = sm.OLS(y, const_X)
            logModel = sm.OLS(y, const_X_log)

            result1 = linearModel.fit()
            result2 = logModel.fit()
            #print result1.summary(), result2.summary()

            #f, (ax1, ax2) = plt.subplots(2,1)
            ##ax1 = plt.subplot()
            #ax1.plot(X, result1.predict(), 'r--', label='lin-fit')
            #ax1.scatter(X, y, label='data')

            #ax2.plot(X_log, result2.predict(), 'g--', label='log-fit')
            #ax2.scatter(X_log, list_control_observed, label='data')

            #ax1.legend()
            #ax2.legend()
            #plt.show()
            #plt.close()
            result2_rsquared_adj.append(result2.rsquared_adj)
            result1_rsquared_adj.append(result1.rsquared_adj)
            var_expected.append(np.var(list_control_expected))

ax = plt.subplot()
ax.scatter(var_expected, result2_rsquared_adj, color='b', label="Log Fits")
ax.scatter(var_expected, result1_rsquared_adj, color='r', label="Linear Fits")
plt.xlabel("Expected PSP variance")
plt.ylabel("$R^2$")
plt.legend()
plt.show()


# print result1.rsquared_adj, result2.rsquared_adj
#ax1.scatter(result1.rsquared_adj, result2.rsquared_adj)
##ax2.scatter(result2.rsquared_adj/result1.rsquared_adj)
#ax2.scatter(i,result1.mse_resid/result2.mse_resid)
#ax1.plot([0, 1], [0, 1], transform=ax1.transAxes)
#ax2.axhline(y=1)
#ax1.set_xlim((0,1))
#ax1.set_ylim((0,1))
#ax1.set_xlabel("Linear")
#ax1.set_ylabel("Log")
#ax2.set_ylabel("Ratio")
#plt.show()
