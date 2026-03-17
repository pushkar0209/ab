import cv2
import os
import sys

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def collect_data(student_id, student_name, num_samples=50):
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', f"{student_id}_{student_name}"))
    create_directory(dataset_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load Haar Cascade for face detection (faster for collection)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    print(f"Collecting data for {student_name} (ID: {student_id}). Please look at the camera.")
    print("Capturing 50 samples...")

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle for visual feedback
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Save the captured face
            # We add a small margin to the face
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            file_name_path = os.path.join(dataset_path, f"{student_id}_{student_name}_{count}.jpg")
            cv2.imwrite(file_name_path, face_img)
            count += 1
            
            # visual feedback
            cv2.putText(frame, f"Captured: {count}/{num_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Face Data Collection', frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data Collection Complete. Samples saved to {dataset_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python data_collection.py <student_id> <student_name>")
        # Interactive mode if arguments are missing
        s_id = input("Enter Student ID: ")
        s_name = input("Enter Student Name: ")
        collect_data(s_id, s_name)
    else:
        collect_data(sys.argv[1], sys.argv[2])
