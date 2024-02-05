import os
import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# Especifica la ruta de la carpeta que contiene tus imágenes
data_folder = "C:/Users/majom/OneDrive/Documentos/Metodologia de la investigacion/Database"

# Listas para almacenar imágenes y etiquetas
images = []
labels = []

# Definir el tamaño de las imágenes
img_height, img_width = 128, 128

# Recorre los archivos en la carpeta
for folder in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, folder)
    if os.path.isdir(folder_path):
        label = 1 if folder.lower() == "aneurysm" else 0  # Asigna etiquetas según la carpeta
        for filename in tqdm(os.listdir(folder_path), desc=f"Loading {folder} images"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_height, img_width))  # Asegura que todas las imágenes tengan el mismo tamaño
            img = img / 255.0  # Normaliza los valores de píxeles entre 0 y 1
            images.append(img)
            labels.append(label)

# Convierte las listas en arrays numpy
X = np.array(images)
y = np.array(labels)

# Aumento de Datos más avanzado
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,  # Rotación adicional
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)

# Validación Cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, val_index in cv.split(X, y):
    X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]

    # Definir la arquitectura de la CNN
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compilar el modelo
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Crear generadores de datos para aumentar el conjunto de entrenamiento
    batch_size = 32
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = train_datagen.flow(X_val, y_val, batch_size=batch_size)

    # Entrenar el modelo
    epochs = 10
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

    # Evaluar el modelo
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print(f'Accuracy en el conjunto de validación: {test_acc}')

    # Realizar predicciones en el conjunto de validación
    predictions = model.predict(X_val)
    y_pred = np.round(predictions)

    # Mostrar métricas de evaluación
    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred))
