import matplotlib.pyplot as plt
import numpy as np
import random

def gradient_regression(X,y, alpha, b, w):
    dw = 0.0
    db = 0.0
    # we make the model by using all the samples
    for i in range(len(X)):
        aux = -2.0*(y[i]-((w*X[i])+b))
        db = db + aux # this solver can easily overflow
        dw = dw + X[i]*aux # this solver can easily overflow
    aux = 1.0/float(len(X))
    b = b - aux*alpha*db
    w = w - aux*alpha*dw
    return b, w

def gradient_regression2(X,y, alpha, b, w):
    aux = -2*(y-(w*X+b)).sum()
    b = b - alpha*aux/float(len(X))
    w = w - alpha*aux/float(len(X))
    return b, w

def plot(fig, X, y, b, w, epochs):
    axs[fig].plot(X, y, 'yo', label='Samples')
    X = np.array(X)
    axs[fig].plot(X, w*X + b, 'k-', label='Regression Loss: ' + '{:9.2f}'.format(loss(X, y, b, w)))
    axs[fig].set_xlabel('{:5.0f}'.format(epochs) + ' epochs')
    axs[fig].legend()
    axs[fig].grid()
    return

def model(X,y,alpha,b,w,epochs):
    fig = 0
    for e in range(epochs):
        b, w = gradient_regression(X, y, alpha, b, w)
        if e % 3000 == 0:
            plot(fig, X, y, b, w, e)
            fig += 1
    return b, w

def prediction(x, b, w):
    return (w*x) + b

def loss(X, y, b, w):
    sum = 0
    for i in range(len(X)):
        sum += (y[i] - prediction(X[i], b, w)) ** 2
    return sum / len(X)

def create_samples(n):
    y = []
    X = list(range(n))
    for i in range(len(X)):
        y.append(20+X[i]+random.random()*20)
    return X, y

def create_samples2(n):
    X = np.array(list(range(n)))
    y = 20+X*np.random.rand(n)*20
    return X, y

X, y = create_samples(40)

# in this example, we do not change epochs'value, since we just show #plots for the values (0, 3000, 6000, 9000)
EPOCHS = 9001

# linear regression evolution
fig, axs = plt.subplots(1, 4, figsize=(15, 4))
# b and w parameters can be better initialized
b , w = model(X, y, 0.001, random.random(), random.random(), EPOCHS)

# linear regression using a wrong alpha value
fig, axs = plt.subplots(1, 4, figsize=(15, 4))
# b an w parameters can be better initialized
model(X, y, 0.0001, random.random(), random.random(), EPOCHS)
y= []
for i in range(len(X)):
    y.append((X[i]-10)**2+random.random()*20*abs(len(X)/2-i))

# loss evolution
fig, axs = plt.subplots(1, 4, figsize=(15, 4))
# b an w parameters can be better initialized
model(X, y, 0.001, random.random(), random.random(), EPOCHS)

plt.show()

print(prediction(10, b, w))
print(prediction(20, b, w))