import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN

class FaceDetector:
    def __init__(self, method='haar', min_face_size=20):
        self.method = method
        self.min_face_size = min_face_size
        
        if self.method == 'haar':
            self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        elif self.method == 'mtcnn':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.detector = MTCNN(keep_all=True, device=device, min_face_size=self.min_face_size)
        else:
            raise ValueError("Method must be 'haar' or 'mtcnn'")

    def detect(self, frame):
        """
        Detect faces in the frame.
        Returns a list of bounding boxes (x, y, w, h) and landmarks (optional/None).
        """
        boxes = []
        
        if self.method == 'haar':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # scaleFactor=1.1, minNeighbors=5 are standard tuning params
            detections = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_face_size, self.min_face_size))
            for (x, y, w, h) in detections:
                boxes.append((x, y, w, h))
                
        elif self.method == 'mtcnn':
            # MTCNN expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections, _ = self.detector.detect(rgb)
            
            if detections is not None:
                for box in detections:
                    if box is not None:
                        x, y, x2, y2 = [int(b) for b in box]
                        w = x2 - x
                        h = y2 - y
                        boxes.append((x, y, w, h))
        
        return boxes
