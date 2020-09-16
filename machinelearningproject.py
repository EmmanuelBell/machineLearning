#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 00:49:14 2020

@author: emmanuelmoudoute-bell
"""

##### Importation des libraires
import numpy as np
import pandas as pds
import Gain

##### Importation de la dataset
dataset = pds.read_excel("udemy.xlsx")


##### Création de notre dataset
X = dataset[["CAC40", "SP500", "DAX30", "EURUSD", "NQ100"]]
y = dataset[["SP500"]]


##### Découpage de la dataset
X_train = X.iloc[0:1000,:].values
y_train = y.iloc[1:1001,:].values

X_test = X.iloc[1001:1476,:].values
y_test = y.iloc[1002:1477,:].values


##### Normalisation des données
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

y_train_sc = sc.fit_transform(y_train)


##### Création et entraînement d'une régression linéaire 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train_sc, y_train_sc)


##### Prédictions
y_pred = lr.predict(X_test_sc)
y_pred = sc.inverse_transform(y_pred)



from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

import matplotlib.pyplot as plt
plt.plot(y_test, color='gray')
plt.plot(y_pred, color='green')

Gain.gain(y_test, y_pred)


from sklearn.svm import SVR

svr = SVR()

svr.fit(X_train_sc, y_train_sc)

y_pred_0 = svr.predict(X_test_sc)
y_pred_0 = sc.inverse_transform(y_pred_0)

Gain.gain(y_test, y_pred_0)

from sklearn.svm import LinearSVR

svr_lin = LinearSVR(max_iter = 100000)

svr_lin.fit(X_train_sc, y_train_sc)

y_pred_1 = svr_lin.predict(X_test_sc)
y_pred_1 = sc.inverse_transform(y_pred_1)

Gain.gain(y_test, y_pred_1)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()

rfr.fit(X_train_sc, y_train_sc)
y_pred_2 = rfr.predict(X_test_sc)
y_pred_2 = sc.inverse_transform(y_pred_2)

gain(y_test, y_pred_1,seuil_hausse = 0.1, seuil_baisse = -0.1, effet_de_levier = 20)