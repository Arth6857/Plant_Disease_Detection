import os
import json
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

working_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = f"{working_dir}/plantvillage dataset/color"
train_path = f"{working_dir}/train_data"
val_path = f"{working_dir}/val_data"

# Load class indices
with open(f"{working_dir}/data/class_indices.json") as f:
    class_indices = {int(k): v for k, v in json.load(f).items()}
    class_names = list(class_indices.values())
    num_classes = len(class_names)

print("Preparing train/val split...")

# Create train/val directories if they don't exist
if not os.path.exists(train_path) or not os.path.exists(val_path):
    print("Creating train/val directory structure...")
    
    # Clean up old directories
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(val_path):
        shutil.rmtree(val_path)
    
    # Create new directories
    os.makedirs(train_path)
    os.makedirs(val_path)
    
    # Split dataset 80/20
    for class_name in class_names:
        train_class_path = os.path.join(train_path, class_name)
        val_class_path = os.path.join(val_path, class_name)
        os.makedirs(train_class_path)
        os.makedirs(val_class_path)
        
        source_class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(source_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle and split
        np.random.seed(42)
        np.random.shuffle(image_files)
        split_idx = int(0.8 * len(image_files))
        
        # Copy to train
        for img_file in image_files[:split_idx]:
            src = os.path.join(source_class_path, img_file)
            dst = os.path.join(train_class_path, img_file)
            shutil.copy(src, dst)
        
        # Copy to val
        for img_file in image_files[split_idx:]:
            src = os.path.join(source_class_path, img_file)
            dst = os.path.join(val_class_path, img_file)
            shutil.copy(src, dst)
        
        print(f"  {class_name}: {len(image_files[:split_idx])} train, {len(image_files[split_idx:])} val")

print("\n✅ Train/Val split complete!")

# Data augmentation
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

val_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
print("\nCreating data generators...")
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

print(f"Train batches: {len(train_generator)}")
print(f"Val batches: {len(val_generator)}")

# Build transfer learning model
print("\nBuilding model with MobileNetV2...")
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

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

print(f"Model parameters: {model.count_params():,}")

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

# ===== PHASE 1: Frozen Base =====
print("\n" + "="*70)
print("PHASE 1: Training with FROZEN MobileNetV2 (15 epochs)")
print("="*70)

base_model.trainable = False
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history1 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ===== PHASE 2: Fine-tune =====
print("\n" + "="*70)
print("PHASE 2: Fine-tuning UNFROZEN layers (10 epochs)")
print("="*70)

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
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

# Final evaluation
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"✅ Final Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Final Validation Loss: {val_loss:.4f}")

print("\n✅ Training complete! Model is ready to use.")
