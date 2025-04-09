<<<<<<< HEAD
# Facial Emotion Recognition using Custom CNN and MobileNetV2 (TFLite Deployed)

This repository provides a lightweight and efficient **Facial Emotion Recognition (FER)** system based on both a **Custom CNN** architecture and a **MobileNetV2 transfer learning model**. Both models are trained on the FER-2013 dataset and converted to **TensorFlow Lite (TFLite)** format for deployment in low-resource environments and real-time applications.

---

## 🔍 Project Highlights

- 📊 **Dataset**: FER-2013 (7 emotion classes)
- 🧠 **Models**:
  - Custom-built CNN
  - Pretrained MobileNetV2 (transfer learning)
- 🛠️ **Technologies**: TensorFlow, TFLite, Keras
- 🚀 **Deployment Ready**: Models converted to `.tflite` format
- 🌐 **Web App**: Simple demo app built with Streamlit (see `app.py`)
- 📁 **Structure**:
  - `src/`: Training code for both models
  - `models/`: Saved `.tflite` models
  - `app.py`: Web interface to test emotion predictions

---

## 📁 Folder Structure
emotion-recognition-app/
│
├── models/
│   ├── custom_cnn_model.tflite
│   └── mobilenetv2_model.tflite
│
├── fer2013_data/             # (optional if dataset too large)
│
├── app.py                    # Streamlit Web App
├── README.md                 # Project Overview
├── requirements.txt          # Python Dependencies
├── src/
│   ├── train_cnn.py          # Custom CNN Training
│   └── train_mobilenet.py    # MobileNetV2 Training
└── LICENSE

# Lightweight Facial Emotion Recognition 🎭

A lightweight and efficient facial emotion recognition system using a custom CNN and MobileNetV2, deployed with TensorFlow Lite and a Streamlit web interface.

## 🚀 Features

- Custom CNN and MobileNetV2-based emotion classification
- TFLite model deployment for efficiency
- Streamlit web app for user interaction
- Training code and evaluation scripts

## 📁 Project Structure

- `models/` — Trained TFLite models
- `src/` — Training scripts
- `app.py` — Streamlit Web App
- `paper/` — Final research paper
- `requirements.txt` — Dependencies

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run app.py

---

## 🧪 Dataset

- **Source**: [FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- **Structure**:
  - `train/`: Training images categorized by emotion
  - `test/`: Test images categorized by emotion
- **Classes**: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`

---

## 🧠 Models

### ✅ Custom CNN
- Lightweight CNN with 3 convolutional blocks
- ~0.7M parameters
- Trained with grayscale images (48x48)

### ✅ MobileNetV2 (Transfer Learning)
- Pretrained on ImageNet
- Fine-tuned for FER on RGB images (48x48)
- Ideal for fast deployment on mobile or edge devices

---

## ⚙️ Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/your-username/emotion-recognition.git
cd emotion-recognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (Optional: models already provided)
python src/train_cnn.py
python src/train_mobilenet.py

# 4. Run Streamlit web app
streamlit run app.py

🤝 Contributing
Feel free to contribute ideas, issues, or improvements via pull requests! Your feedback is always appreciated.

📢 Author
Mohammed Irfan
BTech in AI & Data Science
mi3253050@gmail.com
GITHUB:https://github.com/MdIrfan325
=======
# emotion-recognition-app
Lightweight Facial Emotion Recognition using CNN and MobileNetV2 with Streamlit &amp; TFLite Deployment
>>>>>>> af56b9c4af3a80b0293272646d4f1cdd6475629d
