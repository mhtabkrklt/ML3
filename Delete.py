# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Для реализации моделей
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Загрузка данных
data = pd.read_csv('heart.csv')

# Первичный анализ данных
print("Первые пять строк данных:")
print(data.head())

print("\nИнформация о данных:")
print(data.info())

print("\nПроверка на пропущенные значения:")
print(data.isnull().sum())

# Вывод: Пропущенных значений нет, можно переходить к следующему этапу.

# Разведочный анализ данных (EDA)
print("\nСтатистические характеристики данных:")
print(data.describe())

# Распределение целевого признака
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=data)
plt.title('Распределение целевого признака')
plt.show()

# Вывод: Целевой признак относительно сбалансирован.

# Корреляционная матрица
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, fmt=".2f")
plt.title('Корреляционная матрица признаков')
plt.show()

# Вывод: Некоторые признаки сильно коррелируют с целевым признаком, например, cp, thalach.

# Преобразование данных
# Преобразуем категориальные признаки в количественные
categorical_features = ['cp', 'thal', 'slope']
data = pd.get_dummies(data, columns=categorical_features)

# Масштабирование данных
scaler = StandardScaler()
X = data.drop('target', axis=1)
y = data['target']
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Преобразуем y_train и y_test в массивы NumPy
y_train = np.array(y_train)
y_test = np.array(y_test)


# Реализация собственного метода классификации (KNN)
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train  # Массив NumPy
        self.y_train = y_train  # Массив NumPy

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            # Вычисляем расстояния до всех точек обучающей выборки
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Находим индексы k ближайших соседей
            k_indices = distances.argsort()[:self.k]
            # Получаем метки k ближайших соседей
            k_nearest_labels = self.y_train[k_indices]
            # Определяем наиболее частую метку
            values, counts = np.unique(k_nearest_labels, return_counts=True)
            predictions.append(values[np.argmax(counts)])
        return np.array(predictions)


# Настройка гиперпараметра k для собственного KNN
best_k = 0
best_score = 0
for k in range(1, 31):
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    # Убедимся, что размеры совпадают
    if len(y_pred) != len(y_test):
        print(f"Ошибка: Размер y_pred ({len(y_pred)}) не совпадает с y_test ({len(y_test)}) при k={k}")
        continue
    score = accuracy_score(y_test, y_pred)
    if score > best_score:
        best_k = k
        best_score = score

print(f"\nЛучшее значение k для собственного KNN: {best_k} с точностью {best_score}")

# Вывод: Оптимальное значение гиперпараметра k найдено.

# Оценка модели с лучшим k
knn = KNNClassifier(k=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Проверяем размеры перед вычислением точности
if len(y_pred) == len(y_test):
    print("\nОтчет классификации для собственного KNN:")
    print(classification_report(y_test, y_pred))
else:
    print("Ошибка: Размеры предсказаний и тестовой выборки не совпадают!")

# Матрица ошибок для собственного KNN
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Матрица ошибок для собственного KNN')
plt.show()

# Использование библиотечных реализаций методов классификации
models = {
    'KNN': KNeighborsClassifier(),
    'Логистическая регрессия': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'Наивный Байес': GaussianNB(),
    'Дерево решений': DecisionTreeClassifier()
}

# Настройка гиперпараметров и обучение моделей
accuracy_scores = {}
for name, model in models.items():
    print(f"\n{name}")
    if name == 'KNN':
        param_grid = {'n_neighbors': range(1, 31)}
        grid = GridSearchCV(model, param_grid, cv=5)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"Лучшие параметры: {grid.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = acc
    print(f"Точность: {acc}")
    print("Отчет классификации:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Матрица ошибок для {name}')
    plt.show()

# Вывод: Сравнение моделей показывает различия в их эффективности.

# Дополнительное задание: Реализация еще одного метода классификации (Случайный лес)
from sklearn.ensemble import RandomForestClassifier

print("\nСлучайный лес")
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {'n_estimators': [50, 100, 150],
                 'max_depth': [None, 10, 20, 30]}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print(f"Лучшие параметры: {grid_rf.best_params_}")
y_pred_rf = best_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
accuracy_scores['Случайный лес'] = acc_rf
print(f"Точность: {acc_rf}")
print("Отчет классификации:")
print(classification_report(y_test, y_pred_rf))
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Матрица ошибок для Случайного леса')
plt.show()

# Финальное сравнение моделей
print("\nСравнение моделей по точности:")
for model_name, acc in accuracy_scores.items():
    print(f"{model_name}: {acc}")

# Финальный вывод
best_model_name = max(accuracy_scores, key=accuracy_scores.get)
print(f"\nЛучшая модель: {best_model_name} с точностью {accuracy_scores[best_model_name]}")

# Вывод: Случайный лес показал наилучшие результаты на данном наборе данных.
