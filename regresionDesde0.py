'''
La regresion lineal desarrolla un modelo que ajusta las muestras usando una recta.
y = wx + b
hiperparametros : (w,b)
w: pendiente
b: offset
'''

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1,4, figsize=(15,4))
X = np.array(range(40))
for fig,b,w in zip([0,1,2,3],[20,15,10,5],[0.2,0.4,0.6,0.8]):
    y = b + w*X
    axs[fig].plot(X,y,"b-",label="b="+format(b)+";w="+format(w))
    axs[fig].set_ylim(bottom=0,top=50)
    axs[fig].legend()
    axs[fig].grid()
plt.show()

plt.savefig("EcuacionRecta.png")