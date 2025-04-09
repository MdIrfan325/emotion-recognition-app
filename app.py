import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os

# Emotion labels based on FER-2013
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load TFLite model
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Preprocess the image
def preprocess_image(image):
    image = image.convert("L").resize((48, 48))  # grayscale, resize
    image_array = np.array(image).astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # (48, 48, 1)
    image_array = np.expand_dims(image_array, axis=0)   # (1, 48, 48, 1)
    return image_array

# Run inference
def predict_emotion(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = EMOTION_LABELS[np.argmax(output_data)]
    confidence = np.max(output_data)
    return predicted_label, confidence

# Streamlit UI
st.set_page_config(page_title="Facial Emotion Recognition", layout="centered")
st.title("Facial Emotion Recognition Web App")

# Model selection
model_option = st.selectbox("Select Model", ("Custom CNN", "MobileNetV2"))
model_filename = "custom_cnn_model.tflite" if model_option == "Custom CNN" else "mobilenet_model.tflite"
model_path = os.path.join("models", model_filename)

# Load model
interpreter = load_model(model_path)

# Image upload or webcam
mode = st.radio("Choose input method", ("Upload Image", "Use Webcam"))

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None and interpreter is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        preprocessed = preprocess_image(image)
        label, confidence = predict_emotion(interpreter, preprocessed)

        st.success(f"Prediction: **{label}** ({confidence*100:.2f}% confidence)")

elif mode == "Use Webcam":
    st.info("Click 'Start Camera' to begin live prediction (OpenCV required)")

    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    if run and interpreter is not None:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(gray, (48, 48))
            norm = face.astype("float32") / 255.0
            norm = np.expand_dims(norm, axis=(0, -1))  # (1, 48, 48, 1)

            label, conf = predict_emotion(interpreter, norm)

            # Annotate the frame
            annotated = cv2.putText(frame.copy(), f"{label} ({conf*100:.1f}%)", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        cap.release()

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 40px;
        background-color: #f0f2f6;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        color: #888;
        z-index: 9999;
    }
    </style>
    <div class="footer">
        Developed by <b>Mohammed Irfan</b> | mi3253050@gmail.com
    </div>
    """,
    unsafe_allow_html=True
)
