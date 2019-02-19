from itertools import product
from numpy.random import shuffle
import os

a = [(7,7), (7,1), (7,11), (9,7), (1,7)]
b = [p for p in product(a, repeat=2)]

shuffle(b)
print (b)

xarr, yarr = [], []
outDirectory = './'

for j in b:
     x,y = zip(*j)
     xarr+=x
     yarr+=y

with open(os.path.join(outDirectory , "randX.txt"),'w') as coordFile:
    coordFile.write(','.join( [str(i) for i in xarr] ))

with open(os.path.join(outDirectory , "randY.txt"),'w') as coordFile:
    coordFile.write(','.join( [str(i) for i in yarr] ))
