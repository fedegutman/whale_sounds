# archivo: plot_metrics.py
import numpy as np
import matplotlib.pyplot as plt

acc = np.load("endTrain_accuracies.npy")
loss = np.load("endTrain_lossVector.npy")  # o endTrain_ta_lossVector.npy si existe

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(acc, label="Accuracy")
plt.title("Accuracy por época")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss, label="Loss", color="orange")
plt.title("Loss por época")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()
