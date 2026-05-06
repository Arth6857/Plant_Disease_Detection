#!/usr/bin/env python3
"""
Plant Disease Detection Model - Accuracy Report
Shows model performance metrics clearly for presentations
"""
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from PIL import Image

working_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = f"{working_dir}/plantvillage dataset/color"

# Load model
print("="*80)
print("🌿 PLANT DISEASE DETECTION MODEL - ACCURACY REPORT")
print("="*80)

print("\n📂 Loading model...")
model = tf.keras.models.load_model(f"{working_dir}/model/plant_disease_model.keras")

# Load class indices
with open(f"{working_dir}/data/class_indices.json") as f:
    class_indices = {int(k): v for k, v in json.load(f).items()}
    class_names = list(class_indices.values())

print("✅ Model loaded successfully!")
print(f"📊 Total classes: {len(class_names)}")

# Load test images
print("\n📷 Loading test images...")
images = []
labels = []
true_indices = []

for idx, class_name in class_indices.items():
    class_path = os.path.join(dataset_path, class_name)
    if os.path.exists(class_path):
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Use every 10th image for faster evaluation
        sample_files = image_files[::10][:20]  # Max 20 per class
        
        for img_file in sample_files:
            try:
                img = Image.open(os.path.join(class_path, img_file)).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
                labels.append(idx)
                true_indices.append(idx)
            except:
                pass

images = np.array(images)
true_indices = np.array(true_indices)

print(f"✅ Loaded {len(images)} test images")

# Make predictions
print("\n🔍 Making predictions...")
predictions = model.predict(images, verbose=0)
predicted_indices = np.argmax(predictions, axis=1)
confidence_scores = np.max(predictions, axis=1) * 100

# Calculate metrics
accuracy = accuracy_score(true_indices, predicted_indices)
precision = precision_score(true_indices, predicted_indices, average='weighted', zero_division=0)
recall = recall_score(true_indices, predicted_indices, average='weighted', zero_division=0)
f1 = f1_score(true_indices, predicted_indices, average='weighted', zero_division=0)

# Display results
print("\n" + "="*80)
print("📊 MODEL PERFORMANCE METRICS")
print("="*80)
print(f"✅ Overall Accuracy:     {accuracy * 100:>6.2f}%")
print(f"✅ Weighted Precision:   {precision * 100:>6.2f}%")
print(f"✅ Weighted Recall:      {recall * 100:>6.2f}%")
print(f"✅ Weighted F1-Score:    {f1 * 100:>6.2f}%")
print("="*80)

print(f"\n📈 Average Confidence Score: {np.mean(confidence_scores):.2f}%")
print(f"📈 Min Confidence: {np.min(confidence_scores):.2f}%")
print(f"📈 Max Confidence: {np.max(confidence_scores):.2f}%")

# Show sample predictions
print("\n" + "="*80)
print("🎯 SAMPLE PREDICTIONS (First 10)")
print("="*80)
for i in range(min(10, len(images))):
    true_class = class_names[true_indices[i]].replace("_", " ")
    pred_class = class_names[predicted_indices[i]].replace("_", " ")
    conf = confidence_scores[i]
    status = "✅" if predicted_indices[i] == true_indices[i] else "❌"
    print(f"{status} True: {true_class:40s} | Pred: {pred_class:40s} | Conf: {conf:5.1f}%")

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)
