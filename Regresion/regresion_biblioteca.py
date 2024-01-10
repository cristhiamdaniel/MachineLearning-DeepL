import numpy as np
from sklearn.linear_model import LinearRegression

# Datos y modelo
X, y = np.array([[1], [2], [3], [4], [5]]), np.array([3.1, 5.3, 7.2, 9.8, 11.9])
model = LinearRegression().fit(X, y)

# Imprimir el modelo
print(f"El modelo es: y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}")
