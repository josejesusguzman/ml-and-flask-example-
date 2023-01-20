# Vamos a usar un clasificador de bosque aleatorio
from sklearn.ensemble import RandomForestClassifier
# Importamos bibliotecas
import pandas as pd
import numpy as np

# Cargamos los datos
df = pd.read_csv('titanic.csv')
df.head()

df.drop(["PassengerId","Name","Cabin","Ticket"],axis=1,inplace=True)

print(df.isnull().sum())

df.dropna(subset=['Embarked','Age'],inplace=True)
print(df.isnull().sum())

dummies = pd.get_dummies(df.Sex)
dummies2 = pd.get_dummies(df.Embarked)

new_df = pd.concat([df,dummies,dummies2],axis='columns')
print(type(new_df))

new_df.drop(['Sex','Embarked'],axis='columns',inplace=True)
print(new_df.head(10))

new_df.info()

x = new_df[new_df.columns.difference(['Survived'])]
y = new_df['Survived']

classifier = RandomForestClassifier()
classifier.fit(x, y)

import joblib

joblib.dump(classifier, 'classifier.pkl')
