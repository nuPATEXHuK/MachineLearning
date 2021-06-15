import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

current_path = os.getcwd()
# Файл с данными должен находиться в папке со скиптом
# TODO: лучше будет отвязаться от конкретного названия файла
filename = os.path.join(current_path, "winequality-white.csv")

# TODO: Структура массива, в дальнейшем нужно изменить
min_max_mass = [[], [], [], [], [], [], [], [], [], [], []]

quality_mass = []
characteristic_mass = []
result_csv = []


# Парсинг строк из csv (последний столбец - результат)
def parse_row(row):
    characteristic = []
    quality = None
    for i, value in enumerate(row):
        if len(row) - 1 == i:
            quality = int(value)
        else:
            min_max_mass[i].append(float(value))
            characteristic.append(float(value))
    quality_mass.append(quality)
    characteristic_mass.append(characteristic)


# Функция для стандартизации данных
def standardization(x, min, max):
    return (x - min) / (max - min)


# TODO: NumPy массивы, конкретные цифры нужно будет заменить (разделение обучающей и тестовой выборки)
wine_characteristic_train = np.zeros((4697, 11), dtype=float)
wine_quality_train = np.zeros(4697, dtype=int)
wine_characteristic_test = np.zeros((201, 11), dtype=float)
wine_quality_test = np.zeros(201, dtype=int)

# Читаем сырые данные
with open(filename) as File:
    reader = csv.reader(File, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    i = 0
    for row in reader:
        # Первую строчку не читаем, т.к. там названия столбцов
        if i != 0:
            parse_row(row)
        i += 1

# Стандартизируем данные
for i, value in enumerate(min_max_mass):
    min_mass = np.min(value)
    max_mass = np.max(value)
    for current_row in characteristic_mass:
        current_row[i] = standardization(current_row[i], min_mass, max_mass)

# Заполняем обучающую и тестовую выборку
j = 0
for i, value in enumerate(characteristic_mass):
    if i < 4697:
        wine_characteristic_train[i] = characteristic_mass[i]
        wine_quality_train[i] = quality_mass[i]
    else:
        wine_characteristic_test[j] = characteristic_mass[i]
        wine_quality_test[j] = quality_mass[i]
        result_csv.append("{};".format(quality_mass[i]))
        j += 1

wine_quality_train_cat = keras.utils.to_categorical(wine_quality_train, 10)
wine_quality_test_cat = keras.utils.to_categorical(wine_quality_test, 10)

# Структура НС
# TODO: есть привязка к конкретному количеству данных (11), либо принять как факт и данные подгонять под это,
#       либо сделать без привязки и переобучать сеть на другом количестве данных.
model = keras.Sequential([
    Flatten(input_shape=(11, 1)),
    Dense(500, activation='relu'),
    Dense(500, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Начинаем обучение модели
model.fit(wine_characteristic_train, wine_quality_train_cat, batch_size=100, epochs=50, validation_split=0.2)

# TODO: убрать привязку к тестовому количеству данных
for num in range(201):
    x = np.expand_dims(wine_characteristic_test[num], axis=0)
    res = model.predict(x)
    result_csv[num] += "{}".format(np.argmax(res))

pred = model.predict(wine_characteristic_test)
pred = np.argmax(pred, axis=1)

# Запись результатов в файл
with open("result.csv", mode='w', encoding='utf8') as File:
    for row in result_csv:
        File.write(row + "\n")
