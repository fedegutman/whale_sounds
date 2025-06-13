import torch
import numpy as np
import pandas as pd
from netDefinition import Net
from torch.serialization import add_safe_globals

# Para permitir deserializar modelos con clase Net
add_safe_globals([Net])

# === 1. Cargar modelo entrenado ===
model = torch.load("endTrain_savedNet", map_location="cpu", weights_only=False)
model.eval()

# === 2. Cargar datos de test ===
X_test = torch.load("data/test_processed/tTestData")  # tensor (N, 1, 75, 45)
pNames = np.load("data/test_processed/pNames.npy")    # array (N,)

print(type(X_test), X_test.shape)
print(type(pNames), pNames.shape)

# ⚠️ Validación simple: evitar error de tamaño
n_samples = min(X_test.shape[0], pNames.shape[0])
if X_test.shape[0] != pNames.shape[0]:
    print(f"⚠️ Tamaño mismatch: X_test tiene {X_test.shape[0]} y pNames tiene {pNames.shape[0]}. Ajustando al mínimo común.")

X_test = X_test[:n_samples]
pNames = pNames[:n_samples]

# === 3. Hacer predicción por batches ===
batch_size = 512
preds = []

with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        output = model(batch)
        pred = torch.argmax(output, dim=1).cpu().numpy()
        preds.append(pred)

preds = np.concatenate(preds)


# === 4. Guardar resultados ===
df = pd.DataFrame({
    "Id": pNames,
    "y": preds
})
df.to_csv("test_predictions.csv", index=False)

print("✅ Predicciones guardadas en 'test_predictions.csv'")
