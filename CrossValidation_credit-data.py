# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:33:13 2020

@author: Darllan
"""

import pandas as pd

base = pd.read_csv('credit-data.csv')

base.loc[base.age < 0, 'age'] = 40.92

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean',axis = 0 )
imputer = imputer.fit(previsores[:,1:4])
previsores[:,1:4] = imputer.fit_transform(previsores[:,1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()

resultados = cross_val_score(classificador, previsores, classe, cv = 10)

resultados.mean()
resultados.std()
