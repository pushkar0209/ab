import os
import sys
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from face_recognition_module import FaceRecognizer
import cv2
import numpy as np

def evaluate_model(data_dir):
    """
    Evaluates the model by running recognition on the training data itself (Sanity Check)
    or a separate test set if organized.
    Here we check consistency: Can we recognize the images we have?
    """
    recognizer = FaceRecognizer(embeddings_path='embeddings.pkl')
    if not recognizer.known_embeddings:
        print("Model not trained.")
        return

    total_images = 0
    correct_predictions = 0
    unknown_predictions = 0

    print(f"Evaluating on data in {data_dir}...")

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                path = os.path.join(root, file)
                true_label = os.path.basename(root) # Folder ID_Name
                
                img = cv2.imread(path)
                if img is None: continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # We assume the image IS the face (cropped), or we detect?
                # The data collection saves cropped faces.
                # In recognition module, we usually resize and normalize.
                
                pred_label, dist = recognizer.recognize(img_rgb)
                
                # Check match (Name matches)
                # pred_label is "ID_Name"
                if pred_label == true_label:
                    correct_predictions += 1
                elif pred_label == "Unknown":
                    unknown_predictions += 1
                
                total_images += 1
                if total_images % 10 == 0:
                    print(f"Processed {total_images} images...", end='\r')

    if total_images == 0:
        print("No images found.")
        return

    accuracy = (correct_predictions / total_images) * 100
    print(f"\nResults:")
    print(f"Total Images: {total_images}")
    print(f"Correct: {correct_predictions}")
    print(f"Unknown: {unknown_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    evaluate_model(data_path)
