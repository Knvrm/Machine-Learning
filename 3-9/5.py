import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data5.csv')
dataset.head()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

print ("Матрица признаков")
print(X)
print ("Зависимая переменная")
print(y)

# Новая версия класса-трансформера, предыдущая Imputer - устарела
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X_without_nan = X.copy()
X_without_nan[:, 1:3] = imputer.transform(X[:, 1:3])
X_without_nan

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
print("Зависимая переменная до обработки")
print(y)
y = labelencoder_y.fit_transform(y)
print("Зависимая переменная после обработки")
print(y)

# устаревший подход к использованию OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
labelencoder_X = LabelEncoder()
labelencoder_X.fit_transform(X[:, 0])
X_encoded = X_without_nan.copy()
X_encoded[:, 0] = labelencoder_X.fit_transform(X_encoded[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X_encoded = onehotencoder.fit_transform(X_encoded).toarray()
print("Перекодировка категориального признака")
print(X_encoded)

# создаем копию "грязного" объекта: спропусками и некодированными категориями
X_dirty = X.copy()
X_dirty

# Современный метод трансформации признаков
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# создаем список трансформеров
transformers = [
    ('onehot', OneHotEncoder(), [0]),
    ('imp', SimpleImputer(), [1, 2])
]

# Создаем объект ColumnTransformer и передаем ему список трансформеров
ct = ColumnTransformer(transformers)

# Выполняем трансформацию признаков
X_transformed = ct.fit_transform(X_dirty)
print(X_transformed.shape)
X_transformed

# Можно преобразовать полученный многомерный массив обратно в Dataframe
X_data = pd.DataFrame(
    X_transformed,
    columns=['C1', 'C2', 'C3', 'Age', 'Salary'])
X_data