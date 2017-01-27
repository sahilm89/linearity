import numpy as np
import matplotlib.pyplot as plt

k = 5e-4
c = 9.2
e = np.linspace(0.0001,0.01,1000)
o =  k* ( np.log(e) + c )
ax = plt.subplot(111)
ax.plot(np.log10(e),o, label="$O \propto log(E)$")
ax.plot(np.log10(e), e, label="$O \propto E$")
ax.set_xlabel("Expected (E)", fontsize='large')
ax.set_ylabel("Observed (O) ", fontsize='large')
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
plt.legend(loc="best")
plt.show()
