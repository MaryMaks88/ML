<a name="top"></a>

#### 1. Weight Initialization
    Коли ми щойно створили нейронну мережу, її нейрони ще нічого не знають. 
    У них всередині є Weights (ваги) — це як налаштування гучності для кожного сигналу.
    
    Якщо ми поставимо всі ваги в 0, мережа нічого не вивчить. 
    Якщо поставимо занадто великі числа — вона "вибухне" від перевантаження.
    Weight Initialization — це спосіб дати нейронам правильні випадкові стартові числа 
    (наприклад, за методом Glorot або He), щоб навчання почалося бадьоро.

    he_normal — чудово працює з активацією ReLU.

    glorot_uniform — стандартний вибір (золота середина).

  ```python
    model = keras.Sequential([
        layers.Dense(64, 
                     activation='relu', 
                     kernel_initializer='he_normal', 
                     input_shape=(784,)),
        
        # А тут можна вказати, щоб числа були просто випадковими від -0.05 до 0.05
        layers.Dense(10, 
                     activation='softmax',
                     kernel_initializer=initializers.RandomNormal(stddev=0.01))
    ])
  ```

#### 2. Functional API
    Пам'ятаєш Sequential (потяг)? Так ось, Functional API — це коли ти будуєш не просто лінію, а розгалуження.
    Уяви, що в тебе є два входи (наприклад, фото кота і звук його нявкання) і ти хочеш, 
    щоб мережа об’єднала ці дані. Functional API дозволяє з’єднувати шари як завгодно: 
    паралельно, хрест-навхрест або навіть створювати "зациклені" маршрути.

  ```python
    # functioonal API
    inputs = tf.keras.Input((10,))
    
    output_dense64 = layers.Dense(64, activation='relu')(inputs)
    
    output_dense32 = layers.Dense(32, activation='relu')(inputs)
    
    concat = layers.Concatenate()([output_dense64, output_dense32])
    
    output =  layers.Dense(1, activation="sigmoid")(concat)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
  ```
<img width="1116" height="743" alt="image" src="https://github.com/user-attachments/assets/0fd9a7e6-e361-49b5-9710-614db1c3b070" />

 ```python
    inputs_tabular = tf.keras.Input((10,), name="Tabular_input")
    
    inputs_image =  tf.keras.Input((256, 256), name="Image_input")
    
    flatten_image = layers.Flatten()(inputs_image)
    
    output_dense64 = layers.Dense(64, activation='relu')(flatten_image)
    
    output_dense32 = layers.Dense(32, activation='relu')(inputs_tabular)
    
    concat = layers.Concatenate()([output_dense64, output_dense32])
    
    output_reg =  layers.Dense(1, name="reg_putput")(concat)
    
    output_class = layers.Dense(1, activation="sigmoid", name="clas_output")(concat)
    
    model = tf.keras.Model(inputs=[inputs_tabular, inputs_image], outputs=[output_reg, output_class])
 ```
<img width="1116" height="445" alt="image" src="https://github.com/user-attachments/assets/4cb80a8b-fddb-4143-92f8-31db0c55b83b" />

 ```python

    import tensorflow as tf
    from tensorflow.keras import layers, models, Input, callbacks, initializers
    
    # 1. Вхідні дані (Functional API)
    # Наші цифри — це картинки 28x28 пікселів (всього 784 точки)
    inputs = Input(shape=(784,))
    
    # 2. Перший шар із Weight Initialization
    # Використовуємо 'he_normal', щоб нейрони одразу прокинулися
    x = layers.Dense(128, 
                     activation='relu', 
                     kernel_initializer='he_normal')(inputs)
    
    # 3. Додаємо BatchNormalization & Dropout
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x) # 20% нейронів відпочивають, щоб інші вчилися краще
    
    # 4. Другий шар
    x = layers.Dense(64, activation='relu')(x)
    
    # 5. "Вихід" — 10 варіантів (цифри від 0 до 9)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # 6. Збираємо модель докупи
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # 7. Налаштовуємо Optimizer & Loss
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # 8. Налаштовуємо Callbacks
    # Якщо модель перестане ставати розумнішою, ми зупинимося самі
    stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=3)
 ```


#### 3. Callbacks
    Навчання моделі може тривати довго. Ти ж не хочеш сидіти і дивитися в монітор 3 години?
    Callbacks — це функції-шпигуни, які стежать за процесом. Вони можуть:
    EarlyStopping: Зупинити навчання, якщо модель перестала ставати розумнішою (щоб не витрачати електрику).
    ModelCheckpoint: Автоматично зберігати найкращу версію моделі, поки ти спиш.
    LearningRateScheduler: Змінювати швидкість навчання "на ходу".

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=0.01,
    verbose=1,
    restore_best_weights=True,
)

from tensorflow.keras.callbacks import ModelCheckpoint

check = ModelCheckpoint(
    "{epoch:02d}-{val_loss:.2f}.keras",
    monitor="val_loss",
    save_best_only=True,
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_model.weights.h5",
    save_weights_only=True,
    save_best_only=True
    )

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=4,
    min_lr=1e-5,
    verbose=1,
)

%load_ext tensorboard
log_dir = "logs/fit/"
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
)

callbacks = [early_stop, lr_reduce, checkpoint, tensorboard]

```

[Нагору ↑](#top)
