import pandas as pd

base = pd.read_csv('census.csv')

previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_previsores = LabelEncoder()

previsores[:,1] = label_encoder_previsores.fit_transform(previsores[:, 1])
previsores[:,3] = label_encoder_previsores.fit_transform(previsores[:, 3])
previsores[:,5] = label_encoder_previsores.fit_transform(previsores[:, 5])
previsores[:,6] = label_encoder_previsores.fit_transform(previsores[:, 6])
previsores[:,7] = label_encoder_previsores.fit_transform(previsores[:, 7])
previsores[:,8] = label_encoder_previsores.fit_transform(previsores[:, 8])
previsores[:,9] = label_encoder_previsores.fit_transform(previsores[:, 9])
previsores[:,13] = label_encoder_previsores.fit_transform(previsores[:, 13])

one_hot_encoder = OneHotEncoder(categorical_features = [1,3,5,6,7,8,9,13])
previsores = one_hot_encoder.fit_transform(previsores).toarray()

label_encoder_classe = LabelEncoder()
classe = label_encoder_classe.fit_transform(classe)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25)


from sklearn.neural_network import MLPClassifier
classificador = MLPClassifier(verbose = True, max_iter = 1000, tol = 0.000001)
classificador.fit(previsores_treinamento,classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import accuracy_score, confusion_matrix
precisao = accuracy_score(classe_teste,previsoes)
matriz = confusion_matrix(classe_teste,previsoes)
