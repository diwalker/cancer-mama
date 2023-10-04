from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import cvzone

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("models/keras_model.h5", compile=False)

# Load the labels
class_names = ['benigno', 'maligno', 'normal']

img = cv2.imread('test/test/10314_410335114.png')

# Resize the raw image into (224-height,224-width) pixels
image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

# Normalize the image array
image = (image / 127.5) - 1

# Predicts the model
prediction = model.predict(image)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
texto1 = f"Tipo: {class_name}"
texto2 = f"Taxa de acerto: {str(np.round(confidence_score * 100))[:-2]} %"

print(texto1,texto2)

cvzone.putTextRect(img, texto1,(50,50),scale=2)
cvzone.putTextRect(img, texto2,(50,100),scale=2)


# Listen to the keyboard for presses.
cv2.imshow('Detector de Cancer de Mama', img)
cv2.waitKey(0)