import pandas as pd

base = pd.read_csv('risco-credito2.csv')

previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()

previsores[:,0] = labelEncoder.fit_transform(previsores[:,0])
previsores[:,1] = labelEncoder.fit_transform(previsores[:,1])
previsores[:,2] = labelEncoder.fit_transform(previsores[:,2])
previsores[:,3] = labelEncoder.fit_transform(previsores[:,3])

from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression()
classificador.fit(previsores,classe)

print (classificador.intercept_)
print(classificador.coef_)

resultado = classificador.predict([[0,0,1,2],[3,0,0,0]])

print(resultado)