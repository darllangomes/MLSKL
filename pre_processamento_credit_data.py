# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd

base = pd.read_csv('credit-data.csv')

base.describe()

base.loc [base['age']<0]

#apagar somente os registros com problema
base.drop(base[base.age<0].index, inplace = True)

#preencher os valores com a média

base.mean()

base['age'].mean()

#tirando a média das idades sem levar em consideração
#os registros com valores invalidos
base['age'][base.age>0].mean()

base.loc[base.age <0, 'age']=40.92

#verificar se existe algum elemento nulo
base.loc[pd.isnull(base['age'])]

#base.loc[pd.isnull(base['age']), 'age']= 40.92

#separando atributos previsores do atributo classe
previsores = base.iloc[:,1:4].values
classe = base.iloc[: , 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

imputer = imputer.fit(previsores[:, 0:3])

previsores[:,0:3] = imputer.transform(previsores[:,0:3])



##Colocando os valores na mesma escala
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)