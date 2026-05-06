import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

working_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = f"{working_dir}/plantvillage dataset/color"

print("Loading model...")
# Load the entire model (includes architecture + weights)
model = tf.keras.models.load_model(f"{working_dir}/model/plant_disease_model.keras")

with open(f"{working_dir}/data/class_indices.json") as f:
    class_indices = {int(k): v for k, v in json.load(f).items()}

print("Testing model accuracy...\n")

correct = 0
total = 0

# Test 5 images from each class (quick test)
for idx, class_name in list(class_indices.items())[:10]:  # First 10 classes
    class_path = os.path.join(dataset_path, class_name)
    if os.path.exists(class_path):
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
        
        for img_file in image_files:
            try:
                img = Image.open(os.path.join(class_path, img_file))
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                prediction = model.predict(img_array, verbose=0)
                predicted_idx = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                
                correct += (predicted_idx == idx)
                total += 1
                
                true_class = class_name.replace("_", " ")
                pred_class = class_indices[predicted_idx].replace("_", " ")
                status = "✓" if predicted_idx == idx else "✗"
                print(f"{status} True: {true_class:40s} | Pred: {pred_class:40s} | Conf: {confidence:.1f}%")
            except Exception as e:
                print(f"Error: {e}")

print("\n" + "="*80)
print(f"Quick Test Accuracy: {(correct/total)*100:.1f}% ({correct}/{total} correct)")
print("="*80)
