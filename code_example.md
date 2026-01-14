```python

# імпорт необхідних бібліотек
import numpy as np
import pandas as pd

# створення даних
data = {
    'вага': [140, 130, 150, 170, 160, 180, 120, 200],
    'текстура': [1, 2, 1, 3, 8, 9, 7, 10],
    'таргет': [0, 0, 0, 0, 1, 1, 1, 1]
}

# створення датафрейму з даних
df = pd.DataFrame(data)

# ділимо на X та y
X = df.copy().drop('таргет', axis=1)
y = df['таргет']
X.shape, y.shape

# імпортуємо нашу модель нвчання RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# налаштовуємо гіперпараметри та навчаємо на наших даних
model = RandomForestClassifier(n_estimators=10, max_depth=5, n_jobs=1, random_state=14)
model.fit(X, y)

# створюємо новий об'єкт для тестів і пробуємо передбачити
X_test = pd.DataFrame({'вага': [135], 'текстура': [5]})
y_pred = model.predict(X_test)

# виводимо друком що ж в нас вийшло
if y_pred[0] == 0:
    print('це яблучко')
else:
    print('маємо апельсинчик')

# перевіряємо на які ознади модель звернула найбільше уваги
importances = model.feature_importances_
importances # array([0.15055556, 0.84944444]) - мдель більше уваги звертає на текстуру

```
