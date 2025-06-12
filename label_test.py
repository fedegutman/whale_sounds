import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from glob import glob
from tensorflow.keras import layers, models

# 1. Definimos los paths base
BASE_DIR = r"C:\Users\bianc\Machine\TPFINAL\whale-sound-classification\data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(TRAIN_DIR, "train.csv")

# 2. Función de extracción de features
def extract_features(filename):
    y, sr = librosa.load(filename, sr=1000)      # downsample a 1 kHz
    y = y / np.max(np.abs(y))                    # normalizar amplitud
    
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=32, fmin=50, fmax=500,
        n_fft=200, hop_length=20
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    delta1 = librosa.feature.delta(log_mel, order=1)
    delta2 = librosa.feature.delta(log_mel, order=2)
    
    return np.stack([log_mel, delta1, delta2], axis=-1)  # shape=(32, frames, 3)

# 3. Cargamos etiquetas de train (ajustado a tu CSV)
train_df = pd.read_csv(TRAIN_CSV)  # columnas: 'clip_name' y 'label'

# 4. Extraemos features de todos los clips de train
X_train, y_train = [], []
for _, row in train_df.iterrows():
    clip_name = row['clip_name']      # ahora sí existe
    label     = row['label']
    file_aiff = os.path.join(TRAIN_DIR, clip_name)  # e.g. ".../train/train1.aiff"
    
    feats = extract_features(file_aiff)
    X_train.append(feats)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = np.array(X_train)    # (N_samples, n_mels, n_frames, 3)
y_train = np.array(y_train)

# 5. Reordenamos ejes si es necesario para Keras: (N, height, width, channels)
#    Aquí asumimos que 'n_frames' es constante tras extracción; si no, recorta/pad.
#    Si no necesitas transponer, puedes omitir esta línea.
# X_train = np.transpose(X_train, (0, 2, 1, 3))  

# 6. Definición del modelo CNN
# Definir modelo CNN con GlobalPooling en lugar de Flatten
input_shape = (32, None, 3)  # bandas mel, variable frames, 3 canales
model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(16, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.GlobalAveragePooling2D(),  # reemplaza Flatten()
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 7. Entrenamiento
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# 8. Predicción en test
test_files = sorted(glob(os.path.join(TEST_DIR, "*.aiff")))
preds = []
ids   = []

for file in test_files:
    feats = extract_features(file)
    feats = np.expand_dims(np.transpose(feats, (1, 0, 2)), axis=0)  # (1, height, width, channels)
    print(f"{file} -> shape: {feats.shape}")  # 🔍
    prob = model.predict(feats)[0][0]
    preds.append(prob)

# 9. Guardar CSV de resultados
output = pd.DataFrame({'id': [f.split('/')[-1].replace('.aiff','') for f in test_files],
                       'whale': preds})
output.to_csv('predictions.csv', index=False)

print(f"-- Predicciones guardadas")
