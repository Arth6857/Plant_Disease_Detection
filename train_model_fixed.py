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

print("Loading and preparing dataset...")
print(f"Number of classes: {num_classes}")

# Load all images into memory with proper train/val split
X_train = []
y_train = []
X_val = []
y_val = []

for idx, class_name in class_indices.items():
    class_path = os.path.join(dataset_path, class_name)
    if os.path.exists(class_path):
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Loading {class_name}: {len(image_files)} images")
        
        for i, img_file in enumerate(image_files):
            try:
                img = Image.open(os.path.join(class_path, img_file)).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                # Create one-hot label
                label = np.zeros(num_classes, dtype=np.float32)
                label[idx] = 1
                
                # 80/20 split
                if i % 5 == 0:  # 20% validation
                    X_val.append(img_array)
                    y_val.append(label)
                else:  # 80% training
                    X_train.append(img_array)
                    y_train.append(label)
            except Exception as e:
                print(f"  Error loading {img_file}: {e}")

X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

print(f"\n✅ Dataset Loaded!")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Data augmentation for training only
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

print("\nBuilding Transfer Learning Model...")

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Build full model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

print(f"Model has {model.count_params():,} parameters")

# Callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# ===== PHASE 1: Frozen Base Model =====
print("\n" + "="*70)
print("PHASE 1: Training with FROZEN MobileNetV2 base (20 epochs)")
print("="*70)

base_model.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history1 = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    steps_per_epoch=len(X_train) // 32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ===== PHASE 2: Fine-tune =====
print("\n" + "="*70)
print("PHASE 2: Fine-tuning UNFROZEN layers (15 epochs)")
print("="*70)

base_model.trainable = True

# Freeze early layers, only train last 50 layers
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=15,
    steps_per_epoch=len(X_train) // 32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ===== SAVE MODEL =====
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

model.save_weights(f"{working_dir}/model/model_weights.weights.h5")
print("✅ Model weights saved!")

model.save(f"{working_dir}/model/plant_disease_model.keras")
print("✅ Full model saved!")

# ===== FINAL EVALUATION =====
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"✅ Final Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Final Validation Loss: {val_loss:.4f}")

# Test on a few random samples
print("\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

for i in range(5):
    idx = np.random.randint(0, len(X_val))
    pred = model.predict(np.expand_dims(X_val[idx], axis=0), verbose=0)
    pred_idx = np.argmax(pred[0])
    pred_conf = np.max(pred[0]) * 100
    true_idx = np.argmax(y_val[idx])
    
    true_class = class_names[true_idx].replace("_", " ")
    pred_class = class_names[pred_idx].replace("_", " ")
    status = "✅" if pred_idx == true_idx else "❌"
    
    print(f"{status} True: {true_class:40s} | Pred: {pred_class:40s} | Conf: {pred_conf:.1f}%")

print("\n✅ Training complete! Model is ready to use.")
