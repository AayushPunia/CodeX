import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("sign_model.keras")

# Define class labels (a-z)
class_labels = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z"
]

# Get the expected input shape from the model
input_shape = model.input_shape[1:4]  # (height, width, channels)
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = input_shape

def preprocess_frame(frame):
    """Preprocess the frame to match model input requirements."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # Resize
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)  # Reshape
    return img

def predict_webcam():
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = preprocess_frame(frame)

        # Predict sign language gesture
        prediction = model.predict(img)
        class_idx = np.argmax(prediction)
        label = class_labels[class_idx] if class_idx < len(class_labels) else "Unknown"

        # Display prediction on screen
        cv2.putText(frame, f"Sign: {label}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Sign Language Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the function
predict_webcam()