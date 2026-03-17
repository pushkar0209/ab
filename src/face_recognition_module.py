import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
import cv2
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognizer:
    def __init__(self, embeddings_path='embeddings.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.embeddings_path = embeddings_path
        self.known_embeddings = []
        self.known_names = []
        self.load_embeddings()

    def preprocess(self, face_img):
        """
        Preprocess the face image for the model.
        Resize to 160x160, normalize.
        """
        face_img = cv2.resize(face_img, (160, 160))
        face_img = np.float32(face_img)
        face_img = (face_img - 127.5) / 128.0 # Normalize to [-1, 1]
        # HWC to CHW
        face_img = face_img.transpose(2, 0, 1)
        face_img = torch.tensor(face_img).unsqueeze(0).to(self.device)
        return face_img

    def get_embedding(self, face_img):
        """
        Generate embedding for a single face image.
        """
        face_tensor = self.preprocess(face_img)
        with torch.no_grad():
            embedding = self.resnet(face_tensor).detach().cpu().numpy()
        return embedding.flatten()

    def train(self, data_dir):
        """
        Walk through the data directory, generate embeddings, and save them.
        """
        print("Starting training...")
        embeddings_list = []
        names_list = []

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    path = os.path.join(root, file)
                    # Extract name/ID from folder or filename
                    # Assumption: folder name is "ID_Name"
                    label = os.path.basename(root)
                    
                    try:
                        img = cv2.imread(path)
                        if img is None:
                            continue
                        
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        emb = self.get_embedding(img)
                        embeddings_list.append(emb)
                        names_list.append(label)
                    except Exception as e:
                        print(f"Error processing {path}: {e}")

        # Save
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump((embeddings_list, names_list), f)
        
        self.known_embeddings = embeddings_list
        self.known_names = names_list
        print(f"Training complete. {len(self.known_names)} faces trained.")

    def load_embeddings(self):
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                self.known_embeddings, self.known_names = pickle.load(f)
            print(f"Loaded {len(self.known_names)} embeddings.")
        else:
            print("No embeddings found. Please run training.")

    def recognize(self, face_img, threshold=0.75):
        """
        Recognize a face using Cosine Similarity.
        Returns name, distance (where distance is converted to a positive metric).
        Threshold for cosine similarity is different from Euclidean distance.
        Typically, cosine_similarity > 0.75 is a match for FaceNet.
        """
        if not self.known_embeddings:
            return "Unknown", 0.0

        target_emb = self.get_embedding(face_img).reshape(1, -1)
        known_embs = np.array(self.known_embeddings)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(target_emb, known_embs)[0]
        
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # We can map similarity (1.0 = perfect) to a "distance-like" metric (0.0 = perfect)
        # to remain compatible with main.py's display format "d=0.xx"
        distance = 1.0 - best_similarity

        if best_similarity < threshold:
            return "Unknown", distance
        
        return self.known_names[best_match_idx], distance
