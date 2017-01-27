import glob
import matplotlib.pyplot as plt
import pickle

ax = plt.subplot(111)
for file in glob.glob('/media/sahil/NCBS_Shares_BGStim/patch_data/*/*/plots/*.pkl'):
    print file
    with open(file, 'rb') as input:
        neuron = pickle.load(input)
    for type in ['Control']:
        feature = 0 
        slopeList = []
        sqrList = []
        for numSquares in neuron.experiment[type].keys(): 
            if not numSquares == 1:
                slopeList.append(neuron.experiment[type][numSquares].regression_coefficients[feature]['slope'])
                sqrList.append(numSquares)
    ax.plot(sqrList, slopeList, alpha = 0.5) 
plt.xlabel("Squares")
plt.ylabel("Slopes")
plt.show()
