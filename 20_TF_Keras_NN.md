<a name="top"></a>

### Розгляд наступних тем в нейроних мережах

    TensorFlow2+Keras intro
    Sequential model
    Dense layer
    Activation functions
    Optimizers
    Loss functions
    Dropout
    Batch normalization
    Saving & loading models

    Уяви, що ми будуємо електронний «мозок» для робота. 
    Нейронні мережі — це просто дуже довгі ланцюжки математичних підказок, які допомагають комп'ютеру вчитися на помилках.
    
#### 1. TensorFlow & Keras: Набір конструктора
    TensorFlow — це величезна фабрика з інструментами, 
    а Keras — це як зручна інструкція до LEGO. 
    Замість того, щоб плавити пластик власноруч, ти просто береш готові блоки.

  ```python
  import tensorflow as tf
  from tensorflow import keras
  ```

#### 2. Sequential model: Потяг із вагонів
    Це найпростіший тип моделі. Уяви потяг: дані заходять у перший вагон, 
    проходять крізь нього, потрапляють у другий і так до кінця. 
    Вагони йдуть суворо один за одним.

  ```python
  model = keras.Sequential()
  ```

#### 3. Dense layer: Всі розмовляють з усіма
    Dense (густий шар) — це коли кожен робот у поточному вагоні знайомий з кожним роботом у наступному. 
    Вони передають інформацію всім підряд, щоб нічого не пропустити.

  ```python
  from tensorflow.keras import layers
  # Додаємо шар з 10 "нейронами"
  model.add(layers.Dense(10))
  ```

#### 4. Activation functions: Перемикачі
    Це як фільтр: чи достатньо важлива інформація, щоб передати її далі?
    ReLU — найпопулярніша. Вона каже: "Якщо число від'ємне — зроби його нулем, якщо додатне — залиш як є".

  ```python
  # Додаємо шар з активацією ReLU
  model.add(layers.Dense(10, activation='relu'))
  ```

#### 5. Optimizers: Тренер
    Оптимізатор — це твій тренер. Після кожної спроби він каже: «Так, тут ти помилився,
    наступного разу поверни праворуч на 2 градуси». 
    Він допомагає моделі вчитися швидше. Найпопулярніший — Adam.

#### 6. Loss functions: Лінійка для помилок
    Loss (функція втрат) вимірює, наскільки сильно помилився комп'ютер. 
    Якщо робот мав впізнати кота, а впізнав тостер, функція втрат покаже велике число. 
    Наша мета — звести це число до нуля.

  ```python
  model.compile(optimizer='adam', loss='mean_squared_error')
  ```

#### 7. Dropout: Школа виживання
    Це дуже хитрий трюк. Під час навчання ми навмання «вимикаємо» частину нейронів. 
    Це змушує інші нейрони працювати краще і не сподіватися на «сусіда». 
    Так модель стає самостійнішою.

  ```python
  model.add(layers.Dropout(0.2)) # Вимикаємо 20% нейронів випадково
  ```

#### 8. Batch Normalization: Причісування даних
    Коли дані проходять крізь шари, вони можуть стати занадто великими або занадто малими 
    (як розпатлане волосся). 
    Batch Normalization причісує їх, роблячи однаковими та акуратними, щоб мережі було легше працювати.

  ```python
  model.add(layers.BatchNormalization())
  ```

#### 9. Saving & Loading: Машина часу
    Навчання моделі може тривати години. 
    Щоб не починати спочатку щоразу, ми можемо просто «законсервувати» розум нашого робота у файл.

  ```python
  # Зберігаємо "мозок"
  model.save('my_super_robot.h5')
  
  # Завантажуємо пізніше
  new_model = keras.models.load_model('my_super_robot.h5')
  ```

 ```python
    model = tf.keras.Sequential([
        layers.Input(shape=(24,), batch_size=32),
        layers.Dense(128, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.Dense(64, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.1),
        layers.Dense(1, activation='sigmoid')
        ])
  ```


[Нагору ↑](#top)
