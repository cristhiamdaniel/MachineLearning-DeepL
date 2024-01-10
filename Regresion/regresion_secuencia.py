`import matplotlib.pyplot as plt
import numpy as np
import random

# Parámetros iniciales
EPOCHS = 9001
alpha = 0.001
b = random.random()
w = random.random()
SAMPLES = 40

# Crear muestras
X = np.array(range(SAMPLES))
y = np.array([20 + X[i] + random.random() * 20 for i in range(SAMPLES)])
print("Los datos son: ", X, y)

# Proceso de regresión lineal
fig, axs = plt.subplots(1, 4, figsize=(15, 4))
for epoch in range(EPOCHS):
    dw = 0.0
    db = 0.0
    for i in range(len(X)):
        aux = -2.0 * (y[i] - (w * X[i] + b))
        db += aux
        dw += X[i] * aux
    b -= (1.0 / float(len(X))) * alpha * db
    w -= (1.0 / float(len(X))) * alpha * dw

    if epoch % 3000 == 0 or epoch == EPOCHS - 1:
        axs[epoch // 3000].plot(X, y, 'yo', label='Samples')
        axs[epoch // 3000].plot(X, w * X + b, 'k-', label='Regression Loss: ' + '{:9.2f}'.format(np.mean((y - (w * X + b)) ** 2)))
        axs[epoch // 3000].set_xlabel('{:5.0f}'.format(epoch) + ' epochs')
        axs[epoch // 3000].legend()
        axs[epoch // 3000].grid()

plt.show()

# Mostrar predicciones
print("Prediction for x=10:", (w * 10) + b)
print("Prediction for x=20:", (w * 20) + b)
print("El modelo es: ", w, "* x +", b)`