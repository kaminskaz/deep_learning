import torch
import matplotlib.pyplot as plt

# 1. Load the saved file
checkpoint_path = "./saved_models/xception/xception_test2.pth"
checkpoint = torch.load(checkpoint_path)

# 2. Extract history dictionary
history = checkpoint['history']

epochs = history['epoch']
train_loss = history['train_loss']
val_loss = history['val_loss']
train_acc = history['train_acc']
val_acc = history['val_acc']

# 3. Create the plots
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Val Loss', marker='o')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Train Acc', marker='o')
plt.plot(epochs, val_acc, label='Val Acc', marker='o')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('xception_training_results.png')
plt.show()

print("Plot saved as xception_training_results.png")