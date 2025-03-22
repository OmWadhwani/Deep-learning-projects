# import os
# import cv2
# import numpy as np
# from keras.preprocessing import image
# import warnings
# warnings.filterwarnings("ignore")
# from keras.preprocessing.image import load_img, img_to_array 
# from keras.models import  load_model
# import matplotlib.pyplot as plt
# import numpy as np

# # load model
# model = load_model("best_model.h5")


# face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# cap = cv2.VideoCapture(0)

# while True:
#     ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
#     if not ret:
#         continue
#     gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

#     faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

#     for (x, y, w, h) in faces_detected:
#         cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
#         roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
#         roi_gray = cv2.resize(roi_gray, (224, 224))
#         img_pixels = image.img_to_array(roi_gray)
#         img_pixels = np.expand_dims(img_pixels, axis=0)
#         img_pixels /= 255

#         predictions = model.predict(img_pixels)

#         # find max indexed array
#         max_index = np.argmax(predictions[0])

#         emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
#         predicted_emotion = emotions[max_index]

#         cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     resized_img = cv2.resize(test_img, (1000, 700))
#     cv2.imshow('Facial emotion analysis ', resized_img)

#     if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
#         break

# cap.release()
# cv2.destroyAllWindows
# import tensorflow as tf
# import cv2
# print("TensorFlow Version:", tf.__version__)
# print("OpenCV Version:", cv2.__version__)

# import cv2
# import numpy as np
# import tensorflow
# import tensorflow as tf
# # from tensorflow.keras.models import load_model

# # Load your trained model
# model = tf.saved_model.load("best_model.keras")

# # Define the emotion labels (adjust based on your model's output classes)
# emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# # Initialize the webcam
# cap = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
#     for (x, y, w, h) in faces:
#         face = gray[y:y + h, x:x + w]
#         face = cv2.resize(face, (48, 48))  # Resize to match model input size
#         face = np.expand_dims(face, axis=-1)  # Add channel dimension
#         face = np.expand_dims(face, axis=0)  # Add batch dimension
#         face = face / 255.0  # Normalize
        
#         prediction = model.predict(face)
#         emotion = emotion_labels[np.argmax(prediction)]
        
#         # Draw rectangle and label on frame
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
#     cv2.imshow("Emotion Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array

# # Load Haar cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Load the pre-trained emotion detection model
# model = load_model("best_model.keras")  # Replace with your model file

# # Emotion labels
# emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# # Open the webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to grayscale for face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         # Extract face region (in RGB format)
#         rgb_face = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
#         rgb_face = cv2.resize(rgb_face, (224, 224))  # Resize the RGB face

#         rgb_face = rgb_face.astype("float") / 255.0  # Normalize
#         rgb_face = img_to_array(rgb_face)
#         rgb_face = np.expand_dims(rgb_face, axis=0)  # Add batch dimension

#         # Predict emotion
#         preds = model.predict(rgb_face)[0]
#         emotion = emotion_labels[np.argmax(preds)]

#         # Draw face rectangle
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
#         # Display emotion label
#         cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.8, (0, 255, 0), 2, cv2.LINE_AA)

#     # Show the output frame
#     cv2.imshow("Emotion Recognition", frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

