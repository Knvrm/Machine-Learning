import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

# Загрузка данных
data_path = "C:/test/telecom_churn.csv"
data = pd.read_csv(data_path)
data.head()

# Информаия о признаках набора данных
#data.info()

# Информаия о признаках набора данных
#data['total day minutes'].hist()

# Построение гистограммы с использованием matplotlib
# plt.bar(data.index, data['total day minutes'])
# plt.show()

# Использование matplotlib для представления распределения значений признака
# hist = data['total day minutes'].value_counts()
# plt.bar(hist.index, hist)

# График «ящик с усами» для отдельного признака
#sns.boxplot(data['total day minutes'])

# Использование boxplot для анализа признака для пяти штатов
# top_data = data[['state', 'total day minutes']]
# top_data = top_data.groupby('state').sum()
# top_data = top_data.sort_values('total day minutes', ascending=False)
# top_data = top_data[:5].index.values
# sns.boxplot(y='state', x='total day minutes', data=data[data.state.isin(top_data)], palette='Set3')

# Визуализация распределения признака Churn
#sns.countplot(data['churn'])

# Визуализация пяти популярных штатов
#sns.countplot(data[data['state'].isin(data['state'].value_counts().head(5).index)]['state'])

# Отбор показателей, связанных с затратами клиентов
feats = [f for f in data.columns if 'charge' in f]
#print(feats)

# Диаграммы для сравнения распределения числовых показателей
#data[feats].hist(figsize=(5,5))

# Попарное распределение признаков
#sns.pairplot(data[feats])

# Попарное распределение признаков с визуализацией отказов
#sns.pairplot(data[feats + ['churn']], hue='churn')

# График scatter библиотеки matplotlib
# plt.scatter(data['total day charge'],
#             data['total intl charge'],
#             color="lightblue", edgecolors='blue')
# plt.xlabel('Дневные начисления')
# plt.ylabel('Международн. начисление')
# plt.title('Распределение по 2 признакам')

# Настройка графика: цвет точки зависит от целевого значения признака
# c = data['churn'].map({False: 'lightblue', True: 'orange'})
# edge_c = data['churn'].map({False: 'blue', True: 'red'})
# plt.scatter(data['total day charge'], data['total intl charge'],
#             color=c, edgecolors=edge_c)
# plt.xlabel('Дневные начисление')
# plt.ylabel('Международн. начисление')

# Построение отдельных подмножеств с легендой
# data_churn = data[data['churn']]
# data_loyal = data[~data['churn']]
# plt.scatter(data_churn['total day charge'],
#             data_churn['total intl charge'],
#             color="orange",
#             edgecolors='red',
#             label="Ушли"
#             )
# plt.scatter(data_loyal['total day charge'],
#             data_loyal['total intl charge'],
#             color='lightblue',
#             edgecolors='blue',
#             label="Остались"
#             )
# plt.xlabel('Дневные начисления')
# plt.ylabel('Международн. начисление')
# plt.title('Распределение клиентов')
# plt.legend()
#
# data.corr()
#
# sns.heatmap(data.corr(), cmap=plt.cm.Blues)

plt.show()


