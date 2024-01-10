import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
# import statsmodels.api as sm
from matplotlib.widgets import Slider, TextBox

# Creación del conjunto de datos independiente x
x = np.array(range(1, 11))

# Creación de y1 (distribución lineal)
pendiente = 2
intercepto = 3
ruido = np.random.randn(len(x)) * 2  # Ruido aleatorio
y1 = pendiente * x + intercepto + ruido

# Creación de y2 (distribución cuadrática)
coef_cuadratico = 1
coef_lineal = 2
intercepto = 3
ruido = np.random.randn(len(x)) * 4  # Ruido aleatorio más grande
y2 = coef_cuadratico * x**2 + coef_lineal * x + intercepto + ruido

# Visualización de los datos
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(x, y1, color='blue')
plt.title("Distribución Lineal")
plt.xlabel("x")
plt.ylabel("y1")

plt.subplot(1, 2, 2)
plt.scatter(x, y2, color='red')
plt.title("Distribución Cuadrática")
plt.xlabel("x")
plt.ylabel("y2")

plt.tight_layout()
plt.show()


# Tus datos existentes: x, y1, y2

# Cálculo de la correlación de Pearson
corr_coef_y1, _ = stats.pearsonr(x, y1)
corr_coef_y2, _ = stats.pearsonr(x, y2)

print("Coeficiente de correlación entre x y y1:", corr_coef_y1)
print("Coeficiente de correlación entre x y y2:", corr_coef_y2)

# Visualización
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(x, y1, color='blue')
plt.title(f"Distribución Lineal (Corr: {corr_coef_y1:.2f})")
plt.xlabel("x")
plt.ylabel("y1")

plt.subplot(1, 2, 2)
plt.scatter(x, y2, color='red')
plt.title(f"Distribución Cuadrática (Corr: {corr_coef_y2:.2f})")
plt.xlabel("x")
plt.ylabel("y2")

plt.tight_layout()
plt.show()

"""
# Ajuste de un modelo lineal a los datos (x, y1)
x_sm = sm.add_constant(x)  # Añadir una constante al modelo
modelo_y1 = sm.OLS(y1, x_sm).fit()
residuales_y1 = modelo_y1.resid

# Ajuste de un modelo lineal a los datos (x, y2)
modelo_y2 = sm.OLS(y2, x_sm).fit()
residuales_y2 = modelo_y2.resid

# Función para graficar los residuales
def graficar_residuales(x, residuales, titulo):
    plt.scatter(x, residuales)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(f"Residuales del modelo para {titulo}")
    plt.xlabel("x")
    plt.ylabel("Residuales")
    plt.show()

# Graficar los residuales
graficar_residuales(x, residuales_y1, "y1")
graficar_residuales(x, residuales_y2, "y2")
"""

# Cálculo del coeficiente de correlación de Spearman
spearman_corr_y1, _ = stats.spearmanr(x, y1)
spearman_corr_y2, _ = stats.spearmanr(x, y2)

print("Coeficiente de correlación de Spearman entre x y y1:", spearman_corr_y1)
print("Coeficiente de correlación de Spearman entre x y y2:", spearman_corr_y2)

def calcular_parametros_lineales(x, y):
    n = len(x)
    x_media = np.mean(x)
    y_media = np.mean(y)

    # Calculo de la pendiente (beta_1)
    numerador = sum((x - x_media) * (y - y_media))
    denominador = sum((x - x_media) ** 2)
    beta_1 = numerador / denominador

    # Calculo del intercepto (beta_0)
    beta_0 = y_media - beta_1 * x_media

    return beta_0, beta_1

def predecir_y(x, beta_0, beta_1):
    return beta_0 + beta_1 * x

# Para y1
beta_0_y1, beta_1_y1 = calcular_parametros_lineales(x, y1)
y1_predicha = predecir_y(x, beta_0_y1, beta_1_y1)

# Para y2
beta_0_y2, beta_1_y2 = calcular_parametros_lineales(x, y2)
y2_predicha = predecir_y(x, beta_0_y2, beta_1_y2)

# Cálculo de residuales
residuales_y1 = y1 - y1_predicha
residuales_y2 = y2 - y2_predicha

print("Residuales de y1:", residuales_y1)
print("Residuales de y2:", residuales_y2)

# Función para graficar los residuales
def graficar_residuales(x, residuales, titulo):
    plt.scatter(x, residuales)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(f"Residuales del modelo para {titulo}")
    plt.xlabel("x")
    plt.ylabel("Residuales")
    plt.show()

# Graficar los residuales para y1
graficar_residuales(x, residuales_y1, "y1")

# Graficar los residuales para y2
graficar_residuales(x, residuales_y2, "y2")

# Normalización de los datos para facilitar el entrenamiento
x_normalized = (x - np.mean(x)) / np.std(x)
y1_normalized = (y1 - np.mean(y1)) / np.std(y1)

# Inicialización de parámetros del modelo
beta_0 = 0
beta_1 = 0

# Hiperparámetros
learning_rate = 0.01
epochs = 1000

# Almacenamiento para la visualización de la convergencia
cost_history = []
beta_0_history = []
beta_1_history = []

# Función de costo: Mean Squared Error (MSE)
def compute_cost(beta_0, beta_1, x, y):
    m = len(y)
    total_cost = (1/(2*m)) * np.sum((beta_0 + beta_1 * x - y)**2)
    return total_cost

# Gradiente descendente
for epoch in range(epochs):
    # Cálculo de las predicciones
    y_pred = beta_0 + beta_1 * x_normalized
    
    # Cálculo de los residuales
    residuals = y_pred - y1_normalized
    
    # Cálculo de gradientes
    beta_0_gradient = np.sum(residuals) / len(y1_normalized)
    beta_1_gradient = np.sum(residuals * x_normalized) / len(y1_normalized)
    
    # Actualización de los parámetros
    beta_0 = beta_0 - learning_rate * beta_0_gradient
    beta_1 = beta_1 - learning_rate * beta_1_gradient
    
    # Cálculo del costo y actualización del historial cada 100 épocas
    if epoch % 100 == 0:
        cost = compute_cost(beta_0, beta_1, x_normalized, y1_normalized)
        cost_history.append(cost)
        beta_0_history.append(beta_0)
        beta_1_history.append(beta_1)

# Creación de la figura y los ejes para el gráfico de ajuste de la regresión
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)

# Gráfico inicial de la línea de regresión
line, = ax.plot(x_normalized, beta_0 + beta_1 * x_normalized, color='red')
ax.scatter(x_normalized, y1_normalized, alpha=0.5)
ax.set_title('Ajuste de Regresión Lineal durante el Entrenamiento')
ax.set_xlabel('x Normalizado')
ax.set_ylabel('y1 Normalizado')

# Slider para controlar la visualización de la época
epoch_slider_ax = plt.axes([0.1, 0.05, 0.65, 0.03])
epoch_slider = Slider(epoch_slider_ax, 'Época', 0, 9, valinit=0, valfmt='%0.0f')

# Cuadro de texto para mostrar el valor de la función de costo
loss_textbox_ax = plt.axes([0.1, 0.15, 0.1, 0.05])
loss_textbox = TextBox(loss_textbox_ax, 'Loss', initial=str(cost_history[0]))

# Función de actualización para el Slider
def update(val):
    epoch = int(epoch_slider.val) * 100  # Convertimos a índice de la lista
    current_cost = cost_history[int(epoch_slider.val)]
    line.set_ydata(beta_0_history[int(epoch_slider.val)] + beta_1_history[int(epoch_slider.val)] * x_normalized)
    loss_textbox.set_val("{:.5f}".format(current_cost))
    fig.canvas.draw_idle()

# Registro de la función de actualización al Slider
epoch_slider.on_changed(update)

plt.show()