"""

Cool Logistic Regression Practice

Author: Cora Wu cjw322
Date: 11/13/2018

"""

import numpy as np
import matplotlib.pyplot as plt
from math import log, e

x_values = []
y_values = []
for x in range(21):
    x_values.append(x)
    if x < 10:
        y_values.append(0)
    else:
        y_values.append(1)
plt.scatter(x_values, y_values)
plt.show()

epochs = 400
lambd = 10e-6
learning_rate = .2
epsilon = 10e-8
m = np.random.randn()
b = np.random.randn()


def train():
    global m, b, x_values, y_values

    expected = makePrediction()
    differences = [findLoss(y, prediction) for y, prediction in zip(y_values, expected)]
    dw,db = gradient(expected)
    m -= dw
    b -= db


def gradient(expected):
    weight_derivatives = [(prediction - y)*x for prediction, y, x in zip(expected, y_values, x_values)]
    dw = learning_rate * np.mean(weight_derivatives) - lambd * m

    bias_derivatives = [(prediction - y) for prediction, y in zip(expected, y_values)]
    db = learning_rate * np.mean(bias_derivatives) - lambd * b

    return(dw,db)


def makePrediction():
    #print([1/(1+e**(-1*(m*x+b))) for x in x_values])
    return [1/(1+e**(-1*(m*x+b))) for x in x_values]

def findLoss(y, prediction):
    #assert prediction < 1, "prediction was >= 1"
    return y*log(prediction+epsilon) + (1 - y)*log(1-prediction+epsilon)


for x in range(epochs):
    train()
    if x % 25 == 0:
        expected2 = makePrediction()
        plt.scatter(x_values, expected2)
        plt.scatter(x_values, y_values)
        plt.show()
