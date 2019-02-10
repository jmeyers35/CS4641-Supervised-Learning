import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale, OrdinalEncoder, LabelEncoder


def loadWineData():
    data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
    sep= ';', header=0)
    X, y = data.values[:1201, 0:11], data.values[:1201, 11]

    finalTestX, finalTestY = data.values[1201:, 0:11], data.values[1201:, 11]

    print("Size of wine data: ", len(X))
    return scale(X), y, scale(finalTestX), finalTestY



def loadCar():
    data = pd.read_csv('car.csv')


    X,y = data.values[:1401, 0:6], data.values[:1401, 6]
    finalTestX,finalTestY = data.values[1401:, 0:6], data.values[1401:, 6]

    print("Size of car data: ", len(X))

    enc = OrdinalEncoder()
    enc.fit(X)
    X = enc.transform(X)
    finalTestX = enc.transform(finalTestX)
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    finalTestY = le.transform(finalTestY)
    return X,y, finalTestX, finalTestY
