<a name="top"></a>

<img width="876" height="477" alt="BC681B6B-C4C4-4CC3-ACF2-3245F38EB8E9" src="https://github.com/user-attachments/assets/3d44da5c-72d5-4e78-98bf-12852a6d1dc9" />


    Якщо коротко: ми спочатку зменшуємо кількість "зайвого" в даних, 
    а потім вчимося знаходити ідеальні налаштування для моделей, не витрачаючи на це роки життя.

#### 1. Dimensionality Reduction (t-SNE та UMAP)
    Коли у вас є дані з сотнями характеристик (колонок), ви не можете їх побачити. 
    Зменшення розмірності стискає ці дані до 2D або 3D простору, намагаючись зберегти структуру.
    
    t-SNE (t-distributed Stochastic Neighbor Embedding)
        Цей алгоритм фокусується на локальній структурі. Він намагається зробити так, щоб точки, 
        які були сусідами у багатовимірному просторі, залишалися сусідами і на площині.
        
        Як працює: Він перетворює відстані між точками на ймовірності подібності.
        
        Мінус: Він повільний на великих датасетах і часто "розриває" глобальні зв'язки 
        (ви бачите окремі кластери, але не розумієте, наскільки далеко вони один від одного насправді).

     UMAP (Uniform Manifold Approximation and Projection)
        Це "молодший і швидший" брат t-SNE, побудований на серйозній математичній топології.
        
        Переваги: Він набагато швидший за t-SNE і краще зберігає глобальну структуру (відносини між далекими групами точок).
        
        Результат: Ви отримуєте візуалізацію, де і локальні групи чіткі, і загальна картина має сенс.

```python
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import umap  # pip install umap-learn

# Завантажуємо дані (цифри 0-9, 64 ознаки)
digits = load_digits()
X, y = digits.data, digits.target

# 1. t-SNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

# 2. UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X)
```

#### 2. Bayesian Optimization (Байєсівська оптимізація)
    Уявіть, що ви шукаєте найкращу точку для риболовлі на величезному озері, але кожен закид вудки займає годину.
    Ви не будете кидати навмання. Ви закинете пару разів, а потім будете пробувати там,
    де "клювало" найкраще, або там, де ви ще ні разу не були.
    
    Байєсівська оптимізація робить саме це для підбору параметрів:
    
    Surrogate Model (Сurogatна модель): Вона будує математичну модель 
    (зазвичай Гауссівські процеси), яка "вгадує", як веде себе ваша цільова функція.
    
    Acquisition Function: Вирішує, де спробувати наступного разу — там, 
    де очікується найкращий результат (exploitation), чи там, де ми ще нічого не знаємо (exploration).
    
    Це набагато ефективніше за Grid Search (перебір усіх варіантів), 
    бо алгоритм "вчиться" на кожній спробі.

```python
from bayes_opt import BayesianOptimization # pip install bayesian-optimization

# Функція, яку ми хочемо максимізувати (наприклад, точність моделі)
def black_box_function(x, y):
    return -x**2 - (y - 1)**2 + 1

# Визначаємо межі пошуку
pbounds = {'x': (-2, 2), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(init_points=2, n_iter=10)
print(f"Найкращий результат: {optimizer.max}")
```

#### 3. Hyperparameter Optimization з Optuna
    Optuna — це сучасний фреймворк для автоматизації всього того, про що ми говорили вище.
    Це зараз "золотий стандарт" у Kaggle та індустрії.
    
    Чому вона крута?
    Eager search spaces: Ви описуєте параметри прямо всередині циклу навчання 
    за допомогою звичайного Python (if-else, loops).
    
    SOTA алгоритми: Вона використовує TPÈ (Tree-structured Parzen Estimator) — 
    це варіація байєсівської оптимізації, яка працює дуже швидко.
    
    Pruning (Обрізка): Якщо Optuna бачить, що поточна ітерація навчання 
    показує жахливі результати на перших епохах, вона просто зупиняє її, щоб не витрачати ресурси.

```python
import optuna
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Завантаження даних
data, target = load_breast_cancer(return_X_y=True)
train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)

def objective(trial):
    # Параметри, які Optuna буде підбирати
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
    }

    model = xgb.XGBClassifier(**param)
    model.fit(train_x, train_y)
    
    preds = model.predict(valid_x)
    accuracy = accuracy_score(valid_y, preds)
    return accuracy

# Створення дослідження та запуск
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Найкращі параметри: {study.best_params}")
```

https://medium.com/data-and-beyond/master-the-power-of-optuna-a-step-by-step-guide-ed43500e9b95

[Нагору ↑](#top)
