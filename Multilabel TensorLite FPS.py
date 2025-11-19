import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array

# Load the haarcascade classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the TensorFlow Lite models using tf.lite.Interpreter
emotion_interpreter = tf.lite.Interpreter(model_path='emotion_detection_model_100epochs.tflite')
age_interpreter = tf.lite.Interpreter(model_path='age_model_50epochs.tflite')
gender_interpreter = tf.lite.Interpreter(model_path='gender_model_50epochs.tflite')

# Allocate tensors
emotion_interpreter.allocate_tensors()
age_interpreter.allocate_tensors()
gender_interpreter.allocate_tensors()

# Get input and output details for each model
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

age_input_details = age_interpreter.get_input_details()
age_output_details = age_interpreter.get_output_details()

gender_input_details = gender_interpreter.get_input_details()
gender_output_details = gender_interpreter.get_output_details()

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Male', 'Female']

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    labels = []

    # Start time for FPS calculation
    start_time = cv2.getTickCount()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Prepare image for emotion prediction
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Emotion prediction
        emotion_interpreter.set_tensor(emotion_input_details[0]['index'], roi.astype(np.float32))
        emotion_interpreter.invoke()
        emotion_preds = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])
        emotion_label = class_labels[np.argmax(emotion_preds)]
        label_position = (x, y + h + 85)
        cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Prepare image for gender and age prediction
        roi_color = frame[y:y + h, x:x + w]
        roi_color = cv2.resize(roi_color, (200, 200), interpolation=cv2.INTER_AREA)
        roi_color = np.expand_dims(roi_color, axis=0)

        # Gender prediction
        gender_interpreter.set_tensor(gender_input_details[0]['index'], roi_color.astype(np.float32))
        gender_interpreter.invoke()
        gender_preds = gender_interpreter.get_tensor(gender_output_details[0]['index'])
        gender_label = gender_labels[int(gender_preds[0][0] >= 0.5)]
        gender_label_position = (x, y + h + 25)
        cv2.putText(frame, gender_label, gender_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Age prediction
        age_interpreter.set_tensor(age_input_details[0]['index'], roi_color.astype(np.float32))
        age_interpreter.invoke()
        age_preds = age_interpreter.get_tensor(age_output_details[0]['index'])
        age = round(age_preds[0][0])
        age_label_position = (x, y + h + 55)
        cv2.putText(frame, "Age=" + str(age), age_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Calculate FPS
    end_time = cv2.getTickCount()
    time_taken = (end_time - start_time) / cv2.getTickFrequency()
    fps = 1 / time_taken if time_taken > 0 else 0

    # Display FPS on the frame
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Emotion, Age, and Gender Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
