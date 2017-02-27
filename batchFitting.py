import glob
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.stats as ss
plt.style.use('seaborn-white')
import statsmodels.api as sm

#filelist = glob.glob('/media/sahil/NCBS_Shares_BGStim/patch_data/*/c?/plots/*.pkl')
filelist = ['/media/sahil/InternalHD/c2/plots/c2.pkl'] 
#f, (ax1, ax2) = plt.subplots(1, 2)
control_result2_rsquared_adj = []
control_result1_rsquared_adj = []
control_var_expected = []
gabazine_result2_rsquared_adj = []
gabazine_result1_rsquared_adj = []
gabazine_var_expected = []

tolerance = 5e-4
for i, file in enumerate(filelist):
    print file
    try:
        control_observed = {}
        control_observed_average = []
        gabazine_observed ={}
        gabazine_observed_average = []
        control_expected = {}
        control_expected_average = []
        gabazine_expected ={}
        gabazine_expected_average = []
        feature = 0

        with open(file, 'rb') as input:
            neuron = pickle.load(input)
            for type in neuron.experiment:
                print "Starting type {}".format(type)
                for numSquares in neuron.experiment[type].keys(): 
                    print "Square {}".format(numSquares)
                    if not numSquares == 1:
                        nSquareData = neuron.experiment[type][numSquares]
                        if type == "Control":
                            coords_C = nSquareData.coordwise
                            for coord in coords_C: 
                                if feature in coords_C[coord].feature:
                                    control_observed_average.append(coords_C[coord].average_feature[feature])
                                    control_expected_average.append(coords_C[coord].expected_feature[feature])
                                    control_observed.update({coord: []})
                                    control_expected.update({coord: []})
                                    for trial in coords_C[coord].trials:
                                        if feature in trial.feature:
                                            control_observed[coord].append(trial.feature[feature])
                                            control_expected[coord].append(coords_C[coord].expected_feature[feature])
                        elif type == "GABAzine":
                            coords_I = nSquareData.coordwise
                            for coord in coords_I: 
                                if feature in coords_I[coord].feature:
                                    gabazine_observed.update({coord: []})
                                    gabazine_expected.update({coord: []})
                                    gabazine_observed_average.append(coords_I[coord].average_feature[feature])
                                    gabazine_expected_average.append(coords_I[coord].expected_feature[feature])

                                    for trial in coords_I[coord].trials:
                                        if feature in trial.feature:
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

    if len(control_observed):
        for key in control_observed.keys():
            for element1, element2 in zip(control_observed[key], control_expected[key] ):
                if not (element1 <0 or np.isclose(element1, 0, atol=tolerance) or element2<0 or np.isclose(element2, 0, atol=tolerance)):
                    list_control_observed.append(element1)
                    list_control_expected.append(element2)

    ###########################
            
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
        print result1.summary(), result2.summary()

        #f, (ax1, ax2) = plt.subplots(2,1)
        ##ax1 = plt.subplot()
        #ax1.plot(X, result1.predict(), 'r--', label='lin-fit')
        #ax1.scatter(X, y, label='data')

        #ax2.plot(X_log, result2.predict(), 'g--', label='log-fit')
        #ax2.scatter(X_log, y, label='data')

        #ax1.legend()
        #ax2.legend()
        #plt.show()
        #plt.close()

        control_result2_rsquared_adj.append(result2.rsquared_adj)
        control_result1_rsquared_adj.append(result1.rsquared_adj)
        control_var_expected.append(np.var(list_control_expected))

    if len(list_gabazine_expected)>30 and len(list_gabazine_observed)>30:
        X = np.array(list_gabazine_expected)
        y = np.array(list_gabazine_observed)
        X_log = np.log10(list_gabazine_expected)

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
        #ax2.scatter(X_log, y, label='data')

        #ax1.legend()
        #ax2.legend()
        #plt.show()
        #plt.close()

        gabazine_result2_rsquared_adj.append(result2.rsquared_adj)
        gabazine_result1_rsquared_adj.append(result1.rsquared_adj)
        gabazine_var_expected.append(np.var(list_gabazine_expected))
#plt.hist(list_control_expected)
#plt.show()

    #if len(control_expected_average)>30 and len(control_observed_average)>30:
    #    X = np.array(control_expected_average)
    #    y = np.array(control_observed_average)
    #    X_log = np.log10(control_expected_average)

    #    const_X = sm.add_constant(X)
    #    const_X_log = sm.add_constant(X_log)

    #    #const_X = X
    #    #const_X_log = X_log

    #    linearModel = sm.OLS(y, const_X)
    #    logModel = sm.OLS(y, const_X_log)

    #    result1 = linearModel.fit()
    #    result2 = logModel.fit()
    #    print result1.summary(), result2.summary()

    #    f, (ax1, ax2) = plt.subplots(2,1)
    #    #ax1 = plt.subplot()
    #    ax1.plot(X, result1.predict(), 'r--', label='lin-fit')
    #    ax1.scatter(X, y, label='data')

    #    ax2.plot(X_log, result2.predict(), 'g--', label='log-fit')
    #    ax2.scatter(X_log, y, label='data')

    #    ax1.legend()
    #    ax2.legend()
    #    plt.show()
    #    plt.close()

    #    #gabazine_result2_rsquared_adj.append(result2.rsquared_adj)
    #    #gabazine_result1_rsquared_adj.append(result1.rsquared_adj)
    #    #gabazine_var_expected.append(np.var(list_gabazine_expected))



#f, (ax1, ax2) = plt.subplots(1,2)
#ax1.scatter(control_var_expected, control_result2_rsquared_adj, color='b', label="Log Fits")
#ax1.scatter(control_var_expected, control_result1_rsquared_adj, color='r', label="Linear Fits")
#ax2.scatter(gabazine_var_expected, gabazine_result2_rsquared_adj, color='b', label="Log Fits")
#ax2.scatter(gabazine_var_expected, gabazine_result1_rsquared_adj, color='r', label="Linear Fits")
#
#ax1.set_xlabel("Expected PSP variance")
#ax1.set_ylabel("$R^2$")
#ax2.set_xlabel("Expected PSP variance")
#ax2.set_ylabel("$R^2$")
#
#ax1.set_xlim((min(control_var_expected), max(control_var_expected)))
#ax2.set_xlim((min(gabazine_var_expected), max(gabazine_var_expected)))
#plt.legend()
#plt.savefig('analyzed_temp/variance_R2.svg')
#plt.close()


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
