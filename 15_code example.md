<a name="top"></a>
```python

import numpy as np
import pandas as pd

data_2 = {
    'вага': [140, 130, 150, 170, 160, 180, 120, 200],
    'текстура': [1, 2, 1, 3, 8, 9, 7, 10],
}
new_df = pd.DataFrame(data_2)

from sklearn.cluster import KMeans

# Кажемо моделі: "Знайди мені 3 групи (кластери)"
model = KMeans(n_clusters=3)
model.fit(new_df) # Модель розставляє "капітанів"

# Дивимося, хто в якій групі
print(model.labels_) # [0 0 0 2 2 2 0 1]

from sklearn.metrics import silhouette_score

# Отримуємо оцінку від -1 (погано) до 1 (ідеально)
score = silhouette_score(new_df, model.labels_)
print(f"Оцінка якості: {score}") # Оцінка якості: 0.3848355776213203

from sklearn.mixture import GaussianMixture

# Створюємо модель, яка вміє працювати з "ймовірностями"
gmm = GaussianMixture(n_components=3)
gmm.fit(new_df)

# Дивимося ймовірності для кожної точки
probs = gmm.predict_proba(new_df)
print(probs[:5]) # Покаже шанси для перших 5 точок

#[[1.18184319e-01 8.81815681e-01 0.00000000e+00]
 #[4.03085555e-03 9.95969144e-01 0.00000000e+00]
 #[9.99924421e-01 7.55785569e-05 0.00000000e+00]
 #[1.00000000e+00 3.70823586e-39 0.00000000e+00]
 #[1.00000000e+00 1.03440056e-48 0.00000000e+00]]

# Приклад з ірисками

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

model_error = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=23)
    kmeans.fit(X)
    model_error.append(kmeans.inertia_)

plt.plot(range(1, 11), model_error, marker='o')
plt.title('Метод ліктя')
plt.xlabel('Кількість кластерів')
plt.ylabel('Помилка (Inertia)')
plt.show()
```

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/fd5c4dba-c541-41b5-a475-5dad1ad327f2" />


```python

kmeans = KMeans(n_clusters=3, random_state=23)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')

plt.title('Результат кластеризації ірисів')
plt.show()
```

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/3721bfb5-fc28-4ea8-ac73-2c2032dac429" />


```python
score = silhouette_score(X, y_kmeans)
print(f"Наскільки чітко розділені групи: {score:.2f}") # Наскільки чітко розділені групи: 0.55
# Якщо результат більше 0.5 — це дуже непогано!
```
[Нагору ↑](#top)

