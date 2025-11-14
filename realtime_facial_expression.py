import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
#-----------------------------

#face expression recognizer initialization
import json

# Try loading the model directly from the weights file
# This works for models saved with model.save_weights()
try:
    # First, create a new model with the same architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    
    # Then load the weights
    model.load_weights('facial_expression_model_weights.h5')
    
    # Compile the model
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying alternative loading method...")
    
    # Fallback to the original method if the first one fails
    try:
        with open("facial_expression_model_structure.json", "r") as json_file:
            model_json = json_file.read()
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights('facial_expression_model_weights.h5')
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        print("Model loaded using fallback method!")
    except Exception as e2:
        print(f"Failed to load model: {e2}")
        exit(1)
#-----------------------------

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

while(True):
	ret, img = video_capture.read()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	print(faces) #locations of detected faces
	if(face_cascade):
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
			
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
			detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
			
			# Convert to float32 and normalize to [0,1]
			img_pixels = detected_face.astype('float32') / 255.0
			# Add batch dimension and channel dimension
			img_pixels = np.expand_dims(img_pixels, axis=-1)  # Add channel dimension
			img_pixels = np.expand_dims(img_pixels, axis=0)   # Add batch dimension
			
			predictions = model.predict(img_pixels) #store probabilities of 7 expressions
			
			#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
			max_index = np.argmax(predictions[0])
			
			emotion = emotions[max_index]
			
			#write emotion text above rectangle
			cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
			
			#process on detected face end
			#-------------------------

	cv2.imshow('img',img)

	if cv2.waitKey(1) & 0xFF == ord('q'): # Hit 'q' on the keyboard to quit!
		break

# Release handle to the webcam	
video_capture.release()
cv2.destroyAllWindows()
