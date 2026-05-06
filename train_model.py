import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

working_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = f"{working_dir}/plantvillage dataset/color"

# Load class indices
with open(f"{working_dir}/data/class_indices.json") as f:
    class_indices = {int(k): v for k, v in json.load(f).items()}
    class_names = list(class_indices.values())
    num_classes = len(class_names)

print("Creating data generators...")

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# No augmentation for validation
val_datagen = ImageDataGenerator(rescale=1./255)

# Create generators from directory
print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=class_names,
    shuffle=True,
    seed=42
)

print("Loading validation data...")
val_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=class_names,
    shuffle=False,
    seed=42
)

print(f"\nTraining data: {len(train_generator)} batches")
print(f"Validation data: {len(val_generator)} batches")

# Build transfer learning model
print("\nBuilding Transfer Learning Model with MobileNetV2...")
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model
base_model.trainable = False

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

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Model built with {model.count_params():,} parameters")

# Callbacks
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Phase 1: Train with frozen base model
print("\n" + "="*60)
print("PHASE 1: Training with frozen base model (15 epochs)")
print("="*60)
history1 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=callbacks_list,
    verbose=1
)

# Phase 2: Unfreeze and fine-tune
print("\n" + "="*60)
print("PHASE 2: Fine-tuning with unfrozen layers (15 epochs)")
print("="*60)

base_model.trainable = True

# Freeze early layers, unfreeze deeper layers
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=callbacks_list,
    verbose=1
)

# Save model
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)
model.save_weights(f"{working_dir}/model/model_weights.weights.h5")
print("✅ Model weights saved to model_weights.weights.h5")

# Final evaluation
print("\nFinal Evaluation...")
val_results = model.evaluate(val_generator, verbose=0)
print(f"\n✅ Final Validation Accuracy: {val_results[1] * 100:.2f}%")
print(f"Final Validation Loss: {val_results[0]:.4f}")
