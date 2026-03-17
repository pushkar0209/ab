# Interview & Viva Questions

## Technical Questions

### 1. Why FaceNet instead of LBPH or Eigenfaces?
**Answer**: LBPH and Eigenfaces are traditional holistic methods sensitive to lighting and alignment. FaceNet uses Deep Learning (CNNs) to map faces into a 128-dimensional Euclidean space where distances directly correspond to face similarity. It achieves state-of-the-art accuracy (99.63% on LFW) and is robust to pose and illumination changes.

### 2. What is the Triplet Loss function?
**Answer**: FaceNet is trained using Triplet Loss. It minimizes the distance between an anchor and a positive (same person) while maximizing the distance between the anchor and a negative (different person). 
`L = max(d(a, p) - d(a, n) + margin, 0)`

### 3. How does the Anti-Spoofing work in this system?
**Answer**: We use a texture analysis heuristic based on the Laplacian variance. Real faces have high-frequency details (skin texture), while photos or screens often appear smoother or blurry (low variance). We threshold this variance to detect "liveness". In production, we would add blink detection or IR sensors.

### 4. How do you handle False Positives?
**Answer**: We use a similarity threshold (e.g., 0.6 distance). If the distance is higher, the face is "Unknown". Tuning this threshold balances False Acceptance Rate (FAR) and False Rejection Rate (FRR). We also interpret liveness checks to reject high-confidence spoofery.

### 5. Why use Haar Cascades over MTCNN?
**Answer**: Haar Cascades are extremely fast and work well for frontal faces on CPUs (good for demos). MTCNN is slower but handles multiple angles and occlusions better. We implemented a modular detector to switch between them based on hardware availability.

## System Design Questions

### 1. How does the system prevent duplicate attendance?
**Answer**: The `AttendanceManager` checks the CSV/Database for an existing entry with the same `Student_ID` and current `Date` before appending a new record.

### 2. Is this scalable to 1000 students?
**Answer**: Using linear search (O(N)) for matching is slow for 1000s of users. For scale, we would use a vector database (Faiss or Pinecone) for O(log N) approximate nearest neighbor search and move the recognition to a backend API.

### 3. What are the ethical concerns?
**Answer**: Privacy and consent are paramount. Biometric data (embeddings) should be encrypted. We must ensure the system is not used for unauthorized surveillance.
