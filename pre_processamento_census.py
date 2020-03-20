# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:54:37 2020

@author: Darllan
"""

import pandas as pd
base = pd.read_csv('census.csv')

base.describe()

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_previsores = LabelEncoder()

#labels = labelEncoder_previsores.fit_transform(previsores[:,1])

previsores[:,1] = labelEncoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelEncoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelEncoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labelEncoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labelEncoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labelEncoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelEncoder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labelEncoder_previsores.fit_transform(previsores[:,13])



oneHotEncoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = oneHotEncoder.fit_transform(previsores).toarray()
labelEncoder_classe = LabelEncoder()
classe = labelEncoder_classe.fit_transform(classe)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

previsores = scaler.fit_transform(previsores)



previsores.describe()