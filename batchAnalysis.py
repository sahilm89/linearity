import glob
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.stats as ss
plt.style.use('seaborn-white')
import statsmodels.api as sm

fit1, fit2, fit3, fit4, fit5, fit6, fit_ratio = [],[],[],[],[],[],[]
#ax = plt.subplot(111)
filelist = glob.glob('/media/sahil/NCBS_Shares_BGStim/patch_data/*/c?/plots/*.pkl')
#filelist = ['/media/sahil/InternalHD/170208/c3/plots/c3.pkl']

for file in filelist:
    print file
    try:
        control_observed = {}
        gabazine_observed ={}
        control_expected = {}
        gabazine_expected ={}
        feature = 0

        with open(file, 'rb') as input:
            neuron = pickle.load(input)
            for type in neuron.experiment:
                slopeList, sqrList = [], []
                for numSquares in neuron.experiment[type].keys(): 
                    if not numSquares == 1:
                        nSquareData = neuron.experiment[type][numSquares]
                        if type == "Control":
                            slopeList.append(nSquareData.regression_coefficients[feature]['slope'])
                            coords_C = nSquareData.coordwise
                            sqrList.append(numSquares)
                            for coord in coords_C: 
                                if feature in coords_C[coord].feature:
                                    control_observed.update({coord: coords_C[coord].average_feature[feature]})
                                    control_expected.update({coord: coords_C[coord].expected_feature[feature]})
                        elif type == "GABAzine":
                            slopeList.append(nSquareData.regression_coefficients[feature]['slope'])
                            sqrList.append(numSquares)
                            coords_I = nSquareData.coordwise
                            for coord in coords_I: 
                                if feature in coords_I[coord].feature:
                                    gabazine_observed.update({coord:coords_I[coord].average_feature[feature]})
                                    gabazine_expected.update({coord:coords_I[coord].expected_feature[feature]})

            print "Read {} into variables".format(file)
            list_control_observed   = []  
            list_gabazine_observed  = []
            list_control_expected   = []
            list_gabazine_expected  = []
            tolerance = 1e-4
            for key in gabazine_observed.keys():
                #if not (gabazine_observed[key] <0 or np.isclose(gabazine_observed[key], 0, atol=5e-4) or gabazine_expected[key]<0 or np.isclose(gabazine_expected[key], 0, atol=5e-4)):
                if not (gabazine_observed[key] <0 or np.isclose(gabazine_observed[key], 0, atol=tolerance) or gabazine_expected[key]<0 or np.isclose(gabazine_expected[key], 0, atol=tolerance)):
                    list_gabazine_observed.append(gabazine_observed[key])
                    list_gabazine_expected.append(gabazine_expected[key])

            for key in control_observed.keys():
                #if not (control_observed[key] <0 or np.isclose(control_observed[key], 0, atol=1e-4) or control_expected[key]<0 or np.isclose(control_expected[key], 0, atol=5e-4)):
                if not (control_observed[key] <0 or np.isclose(control_observed[key], 0, atol=tolerance) or control_expected[key]<0 or np.isclose(control_expected[key], 0, atol=tolerance)):
                    list_control_observed.append(control_observed[key])
                    list_control_expected.append(control_expected[key])

            f, ax = plt.subplots(2,2)

            if len(list_gabazine_expected) and len(list_gabazine_observed):
                fit_log_g_e_g_o = ss.linregress(np.log10(list_gabazine_expected), list_gabazine_observed)[2]
                fit_g_e_g_o = ss.linregress(list_gabazine_expected, list_gabazine_observed)[2]
                fit1.append(fit_g_e_g_o)
                fit2.append(fit_log_g_e_g_o)

            if len(list_control_expected) and len(list_control_observed):
                fit_log_c_e_c_o = ss.linregress(np.log10(list_control_expected), list_control_observed)[2]
                fit_c_e_c_o = ss.linregress(list_control_expected, list_control_observed)[2]

                #print  zip(list_control_expected, np.log(list_control_expected))
                X = sm.add_constant(list_control_expected)
                X_log = sm.add_constant(np.log(list_control_expected))

                linearModel = sm.OLS(list_control_observed, X)
                logModel = sm.OLS(list_control_observed, X_log)

                result1 = linearModel.fit()
                result2 = logModel.fit()
                print result1.summary()
                print result2.summary()

                fit3.append(fit_c_e_c_o)
                fit4.append(fit_log_c_e_c_o)
                fit_ratio.append(fit_log_c_e_c_o/fit_c_e_c_o)

            ax[0][0].scatter(np.log10(list_gabazine_expected), list_gabazine_observed, label="Gabazine", c= "g", edgecolor='none')
            ax[0][0].scatter(np.log10(list_control_expected), list_control_observed, label="Control", c= "c", edgecolor='none')
            ax[0][0].set_title("log(E) vs O plot")
            ax[0][0].set_xlabel("log(e)")
            ax[0][0].set_ylabel("o")

            #if len(list_gabazine_observed):
            #    ax[0][0].set_xlim((1.2*min(np.log10(list_gabazine_expected)), 1.2*max(np.log10(list_gabazine_expected))))
            #    ax[0][0].set_ylim((0, 1.2*max(list_gabazine_observed)))
            #elif len(list_control_observed):
            #    ax[0][0].set_xlim((1.2*min(np.log(list_control_expected)), 1.2*max(np.log(list_control_expected))))
            #    ax[0][0].set_ylim((0, 1.2*max(list_control_observed)))
            ax[0][0].legend(loc=2,prop={'size':6})

            #### Expected vs Observed
            #fit_g-e_g-o = zip(['slope', 'intercept', 'r_val', 'p_val', 'stderr'] , ss.linregress(list_gabazine_expected, list_gabazine_observed))
            #fit_c-e_c-o = zip(['slope', 'intercept', 'r_val', 'p_val', 'stderr'] , ss.linregress(list_control_expected, list_control_observed))

            ax[1][0].scatter(list_gabazine_expected, list_gabazine_observed, label="Gabazine", c= "g", edgecolor='none')
            ax[1][0].scatter(list_control_expected, list_control_observed, label="Control", c= "c", edgecolor='none')
            ax[1][0].set_title("E vs O plot")
            ax[1][0].set_xlabel("e")
            ax[1][0].set_ylabel("o")
            ax[1][0].legend(loc=2,prop={'size':6})
            #if len(list_control_observed):
            #    ax[1][0].set_xlim((0,1.2*max(list_control_expected)))
            #    ax[1][0].set_ylim((0, 3.0*max(list_control_observed)))

            list_control_observed   = []  
            list_gabazine_observed  = []
            list_control_expected   = []
            list_gabazine_expected  = []

            for key in list(set(gabazine_observed.keys()) & set(control_observed.keys())):
                list_gabazine_observed.append(gabazine_observed[key])
                list_gabazine_expected.append(gabazine_expected[key])
                list_control_observed.append(control_observed[key])
                list_control_expected.append(control_expected[key])

            control_observed  = np.array(list_control_observed)
            gabazine_observed = np.array(list_gabazine_observed)
            control_expected  = np.array(list_control_expected)
            gabazine_expected = np.array(list_gabazine_expected)

            #### Log gabazine vs Control
            #fit_log-g-o_c-o = zip(['slope', 'intercept', 'r_val', 'p_val', 'stderr'] , ss.linregress(np.log10(gabazine_observed), control_observed))
            #fit_g-o_c-o = zip(['slope', 'intercept', 'r_val', 'p_val', 'stderr'] , ss.linregress(gabazine_observed, control_observed))
            if len(control_observed) and len(gabazine_observed):
                fit_log_g_o_c_o = ss.linregress(np.log10(gabazine_observed), control_observed)[2]
                fit_g_o_c_o = ss.linregress(gabazine_observed, control_observed)[2]

                fit5.append(fit_g_o_c_o)
                fit6.append(fit_log_g_o_c_o)

            ax[0][1].scatter(np.log10(gabazine_observed), control_observed, label="Observed", c= "m", edgecolor='none')
            #ax[0][1].scatter(np.log10(gabazine_expected), control_expected, label="Expected", c= "y", edgecolor='none')
            ax[0][1].set_title("log(G) vs C plot")
            ax[0][1].set_xlabel("log(GABAzine)")
            ax[0][1].set_ylabel("Control")
            ax[0][1].legend(loc=2,prop={'size':6})
            #if len(gabazine_observed):
            #    ax[0][1].set_xlim((1.2*min(np.log10(gabazine_observed)),1.2*max(np.log10(gabazine_observed))))
            #if len(control_observed):
            #    ax[0][1].set_ylim((0, 1.2*max(control_observed)))


            #### Gabazine vs Control
            ax[1][1].scatter(gabazine_observed, control_observed, label="Observed", c= "m", edgecolor='none')
            #ax[1][1].scatter(gabazine_expected, control_expected, label="Expected", c= "y", edgecolor='none')
            ax[1][1].set_title("G vs C plot")
            ax[1][1].set_xlabel("GABAzine")
            ax[1][1].set_ylabel("Control")
            ax[1][1].legend(loc=2,prop={'size':6})
            #if len(gabazine_observed):
            #    ax[1][1].set_xlim((0, 1.2*max(gabazine_observed)))
            #if len(control_observed):
            #    ax[1][1].set_ylim((0, 1.2*max(control_observed)))

            plt.tight_layout()
            f.suptitle("{} {} ".format(neuron.date, neuron.index))
            plt.savefig("analyzed_temp/{}/{}_{}_comparisons.svg".format(neuron.features[feature], neuron.date, neuron.index))
            plt.close()
    except:
        print "Some problem with this file. Check {}! ".format(file)
        continue

print fit1, fit2, fit3, fit4, fit5, fit6, fit_ratio

f, ax = plt.subplots(1,1)
ax.scatter(fit3, fit4, color='c', label = "$R^2$")
ax.set_xlabel("Lin")
ax.set_ylabel("Log")
ax.legend()
f.suptitle("{}".format(feature))
plt.tight_layout()
plt.savefig("analyzed_temp/{}/{}_ratio_scatter.svg".format(neuron.features[feature],feature))
plt.close()

fit1 = np.array(fit1)[~np.isnan(fit1)]
fit2 = np.array(fit2)[~np.isnan(fit2)]
fit3 = np.array(fit3)[~np.isnan(fit3)]
fit4 = np.array(fit4)[~np.isnan(fit4)]
fit5 = np.array(fit5)[~np.isnan(fit5)]
fit6 = np.array(fit6)[~np.isnan(fit6)]
fit_ratio = np.array(fit_ratio)[~np.isnan(fit_ratio)]

print fit1, fit2, fit3, fit4, fit5, fit6, fit_ratio
fit1 =fit1[np.greater(fit1,0)] 
fit2 =fit2[np.greater(fit2,0)]
#fit3 =fit3[np.greater(fit3,0)]
#fit4 =fit4[np.greater(fit4,0)]
fit5 =fit5[np.greater(fit5,0)]
fit6 =fit6[np.greater(fit6,0)]
fit_ratio =fit_ratio[np.greater(fit_ratio,0)]
    
print fit1, fit2, fit3, fit4, fit5, fit6, fit_ratio
f, ax = plt.subplots(1,1)
#ax[0].hist(fit1, histtype='step', color='c', label = "$g_e vs g_o$")
#ax[0].hist(fit2, histtype='step', color='m', label = "$log(g_e) vs g_o$")
#ax[0].legend()

ax.hist(fit3, color='c', label = "$c_e \quad vs \quad c_o$", alpha=0.3)
ax.hist(fit4, color='m', label = "$log(c_e) \quad vs \quad c_o$", alpha=0.3)
ax.legend()


#ax[2].hist(fit5, histtype='step', color='c', label = "$g_o vs c_o$")
#ax[2].hist(fit6, histtype='step', color='m', label = "$log(g_o) vs c_o$")
#ax[2].legend()

f.suptitle("{}".format(feature))
plt.tight_layout()
plt.savefig("analyzed_temp/{}/{}_histogram.svg".format(neuron.features[feature],feature))
plt.close()

#ax.plot(sqrList, slopeList, alpha = 0.5) 
#plt.xlabel("Squares")
#plt.ylabel("Slopes")
#plt.show()
f, ax = plt.subplots(1,1)
ax.hist(fit_ratio, color='c', label = "$Ratio of R^2 log/lin$")
ax.legend()
f.suptitle("{}".format(feature))
plt.tight_layout()
plt.savefig("analyzed_temp/{}/{}_ratio_histogram.svg".format(neuron.features[feature],feature))
plt.close()

print fit3, fit4
f, ax = plt.subplots(1,1)
ax.scatter(fit3, fit4, color='c', label = "$R^2$")
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
ax.set_xlabel("Lin")
ax.set_ylabel("Log")
ax.legend()
f.suptitle("{}".format(feature))
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig("analyzed_temp/{}/{}_ratio_scatter.svg".format(neuron.features[feature],feature))
plt.close()
