import cv2
import numpy as np

# Define the path to the models folder
models_path = "D:/Programming/Python/Face detection/Models/"

# Load the pre-trained models
face_net = cv2.dnn.readNetFromCaffe(
    models_path + "deploy.prototxt",
    models_path + "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

age_net = cv2.dnn.readNetFromCaffe(
    models_path + "age_deploy.prototxt",
    models_path + "age_net.caffemodel"
)

gender_net = cv2.dnn.readNetFromCaffe(
    models_path + "gender_deploy.prototxt",
    models_path + "gender_net.caffemodel"
)

# Define age and gender labels
age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        break

    # Prepare the frame for face detection
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            # Ensure the bounding box is within the frame dimensions
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            # Extract the face ROI
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Prepare the face for age and gender prediction
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), 
                                         (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            
            # Predict gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_labels[gender_preds[0].argmax()]

            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_labels[age_preds[0].argmax()]

            # Draw bounding box and add labels
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Gender and Age Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
