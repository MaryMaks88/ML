<a name="top"></a>
    
    Transfer Learning (Переносне навчання) — це коли ми беремо "мозок" супер-розумної нейромережі, 
    яку величезна компанія (як Google або Microsoft) тренувала місяцями на мільйонах картинок, і просто "позичаємо" її знання.
    
    Уяви, що ти хочеш навчити робота розпізнавати різні види українського борщу. 
    Замість того, щоб вчити його з нуля бачити лінії, кольори та форми, ти береш робота, 
    який вже вміє розпізнавати всю їжу світу, і просто кажеш йому: "Ти вже молодець, тепер просто зосередься на буряку та капусті".
    
    Як це працює (3 кроки):
    Base Model (Фундамент): Ми беремо готову модель (наприклад, MobileNet або ResNet). 
    Вона вже бачила мільйони котів, собак, машин і дерев. Вона — майстер з розпізнавання форм.
    
    Freezing (Заморозка): Ми "заморожуємо" основні знання моделі. 
    Ми кажемо: "Не чіпай те, що ти вже знаєш про лінії та кольори, просто передавай мені результати".
    
    New Head (Нова голова): Ми відрізаємо старий "вихід" моделі (який впізнавав 1000 об'єктів) 
    і приклеюємо свій власний, який впізнає лише те, що потрібно нам (наприклад, "Борщ" або "Не борщ").

```python
  from tensorflow.keras.applications import MobileNetV2
  from tensorflow.keras import layers, models
  
  # 1. Беремо "розумну" модель від Google без її "фінальної голови"
  base_model = MobileNetV2(input_shape=(160, 160, 3),
                           include_top=False, # Відрізаємо стару голову
                           weights='imagenet') # Беремо знання з бази ImageNet
  
  # 2. Заморожуємо знання (щоб не зіпсувати їх)
  base_model.trainable = False
  
  # 3. Будуємо нову модель на цьому фундаменті
  model = models.Sequential([
      base_model,
      layers.GlobalAveragePooling2D(), # Перетворюємо картинку в набір ознак
      layers.Dense(1, activation='sigmoid') # Наш фінальний вихід (Так/Ні)
  ])
  
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
VGG16

```python
  vgg16  = tf.keras.applications.VGG16(
       include_top=False,
       input_shape=(128, 128, 3)
  )
  vgg16.trainable = False
  
  model = tf.keras.Sequential([
      tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),
      vgg16,
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(1, activation="sigmoid")
  ])
  
  model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=['accuracy']
  )
```

Розморозка тільки після першого навчання !!!

```python
  base_model.trainable = False  # Заморозили все
  # ... тут код навчання model.fit() на 5-10 епох ...

  # 1. Дозволяємо базовій моделі вчитися
  base_model.trainable = True
  
  # 2. Але скільки шарів у нас в моделі? Давай подивимось
  print("Кількість шарів у базі:", len(base_model.layers))
  
  # 3. Заморожуємо всі шари, КРІМ останніх, наприклад, 20-ти
  fine_tune_at = 130 # Наприклад, якщо всього 150 шарів
  
  for layer in base_model.layers[:fine_tune_at]:
      layer.trainable = False

  model.compile(
      # Ставимо швидкість у 10 разів меншу, ніж зазвичай
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), 
      loss='binary_crossentropy',
      metrics=['accuracy']
  )
  
  # Тепер вчимо ще кілька епох
  model.fit(train_data, epochs=5)
```

[Нагору ↑](#top)
