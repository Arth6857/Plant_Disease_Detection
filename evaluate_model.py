import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

working_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = f"{working_dir}/plantvillage dataset/color"

# Load model
def load_model():
    model = models.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(38, activation='softmax')
    ])
    model.load_weights(f"{working_dir}/model/model_weights.weights.h5")
    return model

# Load class indices
with open(f"{working_dir}/data/class_indices.json") as f:
    class_indices = {int(k): v for k, v in json.load(f).items()}
    class_names = list(class_indices.values())

print("Loading model...")
model = load_model()

print("Loading dataset...")
images = []
labels = []
true_indices = []

# Load images from dataset
for idx, class_name in class_indices.items():
    class_path = os.path.join(dataset_path, class_name)
    if os.path.exists(class_path):
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Loading {class_name}: {len(image_files)} images")
        
        for img_file in image_files[:50]:  # Limit to 50 per class for speed
            try:
                img = Image.open(os.path.join(class_path, img_file))
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(idx)
                true_indices.append(idx)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")

images = np.array(images)
true_indices = np.array(true_indices)

print(f"\nTotal images loaded: {len(images)}")

# Make predictions
print("\nMaking predictions...")
predictions = model.predict(images)
predicted_indices = np.argmax(predictions, axis=1)

# Calculate metrics
accuracy = accuracy_score(true_indices, predicted_indices)
precision = precision_score(true_indices, predicted_indices, average='weighted', zero_division=0)
recall = recall_score(true_indices, predicted_indices, average='weighted', zero_division=0)
f1 = f1_score(true_indices, predicted_indices, average='weighted', zero_division=0)

print("\n" + "="*50)
print("MODEL ACCURACY REPORT")
print("="*50)
print(f"Overall Accuracy:  {accuracy * 100:.2f}%")
print(f"Weighted Precision: {precision * 100:.2f}%")
print(f"Weighted Recall:   {recall * 100:.2f}%")
print(f"Weighted F1-Score: {f1 * 100:.2f}%")
print("="*50)

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(true_indices, predicted_indices, target_names=class_names, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(true_indices, predicted_indices)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{working_dir}/confusion_matrix.png", dpi=150)
print("\n✅ Confusion matrix saved to: confusion_matrix.png")
