 
import  matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import log
x = np.array([0,3,6])
y=np.array([0,30,100])
#colors =np.array([0,30,40])
#plt.scatter(x,y,marker="*",c=colors)
#plt.colorbar()
#plt.show()
#d=pd.date_range("20130206",periods=6)
d=np.around([3.12345,4.55231,-3.664],1)

nplog = np.frompyfunc(log,2,1)
d=nplog(100,15)
print(d)
d=nplog([2,3,4,5,6],2)
print(d)
d=np.log2([2,3,4,5,6])
print(d)
print(type(np.add) == np.ufunc)
def myadd(x,y):
    return np.add(x,y)
myadd1 = np.frompyfunc(myadd,2,1)
print(type(myadd1) == np.ufunc)
d=np.lcm.reduce(np.array([3,6,9]))
print(d)
d=np.gcd.reduce(np.array([3,6,9]))
print(d)
print(np.hypot(3,4))
print(np.hypot([3,4,5],[4,5,6]))
print(np.lcm.reduce([3,6,9]))


