import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

working_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = f"{working_dir}/plantvillage dataset/color"

# Load class indices
with open(f"{working_dir}/data/class_indices.json") as f:
    class_indices = {int(k): v for k, v in json.load(f).items()}
    class_names = list(class_indices.values())
    num_classes = len(class_names)

print("⚡ FAST TRAINING MODE - Loading subset of data...")
print(f"Number of classes: {num_classes}\n")

X_train = []
y_train = []
X_val = []
y_val = []

# Load only 200 samples per class max (instead of all) for fast training
MAX_SAMPLES_PER_CLASS = 200

for idx, class_name in class_indices.items():
    class_path = os.path.join(dataset_path, class_name)
    if os.path.exists(class_path):
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Limit to MAX_SAMPLES_PER_CLASS
        image_files = image_files[:MAX_SAMPLES_PER_CLASS]
        
        print(f"Loading {class_name}: {len(image_files)} images")
        
        for i, img_file in enumerate(image_files):
            try:
                img = Image.open(os.path.join(class_path, img_file)).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                label = np.zeros(num_classes, dtype=np.float32)
                label[idx] = 1
                
                # 80/20 split
                if i % 5 == 0:
                    X_val.append(img_array)
                    y_val.append(label)
                else:
                    X_train.append(img_array)
                    y_train.append(label)
            except Exception as e:
                pass

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

print(f"\n✅ Data loaded!")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True
)

print("\nBuilding MobileNetV2 model...")

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Lightweight custom layers
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

print(f"Parameters: {model.count_params():,}")

# Freeze base model - only train top layers
base_model.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

print("\n" + "="*70)
print("⚡ PHASE 1: Fast training with frozen base (8 epochs)")
print("="*70)

history1 = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=64),
    epochs=8,
    steps_per_epoch=len(X_train) // 64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

print("\n" + "="*70)
print("⚡ PHASE 2: Brief fine-tuning (4 epochs)")
print("="*70)

# Unfreeze last 30 layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=64),
    epochs=4,
    steps_per_epoch=len(X_train) // 64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# Save
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

model.save_weights(f"{working_dir}/model/model_weights.weights.h5")
print("✅ Weights saved!")

model.save(f"{working_dir}/model/plant_disease_model.keras")
print("✅ Model saved!")

# Evaluate
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n✅ Final Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Final Validation Loss: {val_loss:.4f}")

print("\n✅ ⚡ TRAINING COMPLETE!")
