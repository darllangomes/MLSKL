import Orange

#biblioteca Orange realiza os pré processamentos que seriam necessários
#serem feitos no scikit learn


#carregando o dataframe na variável base
base = Orange.data.Table('credit-data.csv')

#visualizando a base
base.domain

#A biblioteca Orange necessita que você modifique o arquivo csv para definir
#qual será o atributo classe, para isso, abra o csv no próprio spyder
# e altere o nome do atributo para c#nome_da_classe


#No caso do atributo do tipo ID, é desnecessário levar ele em consideração
#Utilize o i#nome_do_atributo  para poder ignorar.


#dividindo a base em base de treinamento e de teste.
base_dividida= Orange.evaluation.testing.sample(base,n=0.25)

len(base_dividida[1])

base_treinamento=base_dividida[1]
base_teste= base_dividida[0]

cn2_learner = Orange.classification.rules.CN2Learner()
#aqui é onde, de fato, estão sendo geradas as regras.
classificador = cn2_learner(base_treinamento)

for regras in classificador.rule_list:
    print(regras)
    
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [classificador])
print(Orange.evaluation.CA(resultado))