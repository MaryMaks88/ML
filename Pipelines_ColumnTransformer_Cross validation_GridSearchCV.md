<a name="top"></a>
### Pipelines (Конвеєр)
    Раніше ти робила все по черзі: спочатку масштабувала дані (scaler), потім вчила модель.
    Це як мити яблуко руками, а потім вручну його чавити. Pipeline — це конвеєрна стрічка. 
    Ти кладеш на неї "брудне" яблуко (сирі дані),
    а на виході отримуєш готовий сік (прогноз). Робот сам помиє, почистить і вичавить у правильному порядку.

### ColumnTransformer (Розподільник)
    Уяви, що на твій завод привозять яблука і... пластикові пляшки. Пляшки мити не треба, а яблука — треба. ColumnTransformer каже:
    "Ці колонки (числа) відправ на MinMaxScaler, а ці колонки (наприклад, колір куща) — просто залиш як є або оброби інакше".
    Він дозволяє застосовувати різні правила до різних стовпчиків одночасно.

### Cross-validation (Екзамен з багатьма варіантами)
    Пам’ятаєш, як ми раділи помилці 0 при $k=1$, але це виявилося "списуванням"?
    Cross-validation — це коли ми ділимо наш підручник на 5 частин.
    Спочатку ми вчимося на частинах 1-4 і перевіряємо себе на 5-й. Потім вчимося на 2-5 і перевіряємо на 1-й.
    І так 5 разів.Це гарантує, що робот не просто зазубрив відповіді, а реально зрозумів тему.

### GridSearchCV (Супер-перебирач)
    Ти вручну міняла $k=1, 2, 3, 5$. Це довго!GridSearchCV — це робот-помічник, якому ти даєш список:
    "Спробуй сусідів від 1 до 10 і спробуй два види скалерів".
    Він сам запустить 20 експериментів, порівняє результати і скаже: "Ось найкращий рецепт!".

\\\\\\Code\\\\\\\

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score

data = {
    "висота_куща": [50, 80, 150, 40, 120, 200, 45, 110],
    "вік_куща": [1, 2, 5, 1, 4, 8, 1, 3],
    "кількість_ягід": [20, 45, 120, 15, 100, 250, 18, 90]
}

df = pd.DataFrame(data)

X = df[["висота_куща", "вік_куща"]]
y = df["кількість_ягід"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["висота_куща", "вік_куща"])
])

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', KNeighborsRegressor())
])

param_grid = {'model__n_neighbors': [1, 2, 3, 4, 5]}

grid_search = GridSearchCV(
    pipe,
    param_grid,
    cv=3,
    n_jobs=-1,
    scoring='r2'
)

grid_search.fit(X, y)

print(f"Найкраща кількість сусідів: {grid_search.best_params_['model__n_neighbors']}")
print(f"Найкраща оцінка (R2): {grid_search.best_score_:.2f}")
```
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

```python
num_features = ["Age", "Salary"]
cat_features = ["Sex", "City"]

custom_features = ["Age", "Experience"]

X = df[num_features + cat_features + ["Experience"]].copy()
X = df.drop(columns="Target")
y = df["Target"]


numeric_pipeline = Pipeline([
     ("imputer", SimpleImputer(strategy='median')),
     ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("one_hot", OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop="if_binary"))
])


custom_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='median')),
    ("feature_extractor", FunctionTransformer(func=create_exp_ratio, validate=True)),
    ("scaler", StandardScaler())
])


preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_features),
    ("cat", categorical_pipeline, cat_features),
    ("custom", custom_pipeline, custom_features)
])

full_pipeline = Pipeline([
    
    ("preprocessor", preprocessor),
    ("model", KNeighborsRegressor(2))
])
full_pipeline

new_pred = grid_search.predict(new_df)
print(f"Фінальний прогноз від найкращої моделі: {new_pred[0]:.0f} ягід")
```
[Нагору ↑](#top)
