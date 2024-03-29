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
images, labels = [], []

# Definir el tamaño de las imágenes
img_height, img_width = 128, 128

def load_images_from_folder(folder, label):
    for filename in tqdm(os.listdir(folder), desc=f"Loading {label} images"):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_height, img_width))
        img = img / 255.0
        images.append(img)
        labels.append(label)

# Recorre los archivos en la carpeta
for folder in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, folder)
    if os.path.isdir(folder_path):
        label = 1 if folder.lower() == "aneurysm" else 0
        load_images_from_folder(folder_path, label)

# Convierte las listas en arrays numpy
X, y = np.array(images), np.array(labels)

# Aumento de Datos más avanzado
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)

# Validación Cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Obtener índices de entrenamiento y validación para cada iteración de validación cruzada
for train_index, val_index in cv.split(X, y):
    X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]

    # Definir la arquitectura de la CNN
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

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
