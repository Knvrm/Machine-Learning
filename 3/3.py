import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Добавление шапки DataFrame по умолчанию
data_source = 'iris.data'
d = pd.read_table(data_source, delimiter=',',
                  header=None,
                  names=['sepal_length', 'sepal_width',
                         'petal_length', 'petal_width', 'answer'])
d.head()

# Построение графика с указанием признака отдельных классов
#sb.pairplot(d, hue='answer')
#plt.show()

X_train = d[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = d['answer']
K = 3
# Создание и настройка классификатора
knn = KNeighborsClassifier(n_neighbors=K)
# построение модели классификатора (процедура обучения)
knn.fit(X_train, y_train)

# Использование классификатора
# Объявление признаков объекта
X_test = np.array([[1.2, 1.0, 2.8, 1.2]])
# Получение ответа для нового объекта
target = knn.predict(X_test)
print(target)

from sklearn.model_selection import train_test_split

X_train, X_holdout, y_train, y_holdout = \
    train_test_split(d[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']],
                     d['answer'],
                     test_size=0.3,
                     random_state=17)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_holdout)

accur = accuracy_score(y_holdout, knn_pred)

# Оценка точности классификатора с использованием методики hold-out
print('accuracy:', accur)

# Значения параметра K
k_list = list(range(1, 50))

cv_scores = []
# В цикле проходим все значения параметра K
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, d.iloc[:, 0:4], d['answer'], cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Вычисляем ошибку
MSE = [1 - x for x in cv_scores]

# Строим график
plt.plot(k_list, MSE)
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Mean Squared Error (MSE)')
plt.show()

# Ищем минимум
k_min = min(MSE)

# Пробуем найти прочие минимумы
all_k_min = [k_list[i] for i in range(len(MSE)) if MSE[i] <= k_min]

# Печатаем все оптимальные К для данной модели
print('Optimal Neighbors K: ', all_k_min)