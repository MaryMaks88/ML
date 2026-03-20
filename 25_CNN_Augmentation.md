<a name="top"></a>

#### CNN (Convolutional Neural Networks)
    CNN (Згорткові нейронні мережі) створені спеціально для того, щоб розуміти зображення. 
    Замість того, щоб дивитися на всю картинку одразу і плутатися в мільйонах пікселів, 
    CNN розбиває її на маленькі шматочки.

    Уяви, що ти дивишся на фото кота через маленьке квадратне віконце (фільтр). 
    Ти рухаєш це віконце по всій картинці:
    
    Спочатку помічаєш лінії та кути (вушка, вуса).
    
    Потім бачиш візерунки (смужки на шерсті).
    
    І нарешті розумієш цілий об'єкт (це кіт!).

    Головні деталі CNN:
    Convolution (Згортка): Це те саме віконце-фільтр. Воно "ковзає" по картинці та шукає знайомі ознаки.
    
    Pooling (Пулінг): Це спосіб зменшити картинку. Уяви, що ти дивишся на фото здалеку — деталі зникають, 
    але головна форма залишається. Це допомагає комп'ютеру працювати швидше.
    
    Flatten (Вирівнювання): Коли ми знайшли всі ознаки (вуса, хвіст), ми витягуємо їх у довгу лінію, 
    щоб передати в звичайний шар (Dense), який прийме фінальне рішення.

```python
    from tensorflow.keras import layers, models

    model = models.Sequential([
    # Шукаємо 32 різні ознаки віконцем 3x3
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # Зменшуємо картинку вдвічі
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
  ])
```

#### Data Augmentation (Збільшення даних)
    Уяви, що ти вчиш дитину впізнавати собаку, але показуєш тільки фото золотистого ретривера, 
    який стоїть зліва. Якщо дитина побачить таксу, що біжить справа, вона може її не впізнати.
    
    Нейронні мережі мають ту саму проблему — їм потрібно БАГАТО різних прикладів. 
    Data Augmentation — це спосіб обманути мережу і зробити з однієї картинки десять нових.
    
    Як ми "мучимо" картинку:
    Поворот: Нахиляємо кота на 20°.
    
    Дзеркало: Віддзеркалюємо собаку зліва направо.
    
    Зум: Наближаємо ніс або віддаляємо хвіст.
    
    Яскравість: Робимо фото темнішим або світлішим.
    
    Для комп'ютера це виглядає як абсолютно нові фотографії, і він вчиться значно краще.
    
    Як це зробити в Keras:
    Ми просто додаємо спеціальні шари на самому початку моделі:

```python
  layers.RandomFlip("horizontal"), # Крутимо вліво-вправо
  layers.RandomRotation(0.1),       # Трохи повертаємо
  layers.RandomZoom(0.1),           # Наближаємо
```
<img width="1116" height="743" alt="image" src="https://github.com/user-attachments/assets/0fd9a7e6-e361-49b5-9710-614db1c3b070" />

```python
model = tf.keras.Sequential([

    layers.Input(shape=(32, 32, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(10, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=1,
    restore_best_weights=True
)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[early_stop], epochs=100)

```

    Без Data Augmentation модель може стати "зазубрювачем" (це називається overfitting). 
    Вона вивчить кожну цятку на твоїх 100 фотографіях, але на новому фото з інтернету повністю провалиться. 
    Аугментація робить її "гнучкою" та розумною.

[Нагору ↑](#top)
