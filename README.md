# GestureTalk: Real-Time Sign Language Communication System

## 📌 Overview
GestureTalk is a **real-time sign language communication system** that enables two-way communication between deaf and hearing users.  
It uses **YOLO** and **DWpose** for gesture recognition, **3D avatar animation** for visual representation, and integrates **speech recognition** and **NLP** for smooth interaction.

## ✨ Features
- Real-time gesture recognition using YOLO & DWpose
- 3D avatar animation for realistic sign representation
- Speech-to-sign and sign-to-speech conversion
- NLP-based text processing for improved accuracy
- Optimized with OpenCV, PyTorch, and real-time inference pipelines

## 🛠 Tech Stack
- **Languages:** Python
- **Frameworks & Libraries:** PyTorch, OpenCV, Hugging Face Transformers, NLTK
- **Models:** YOLO, DWpose
- **APIs:** Google Speech-to-Text, Text-to-Speech

## 📂 Project Structure
GestureTalk-Sign-Language-Recognition/
│
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   ├── sample_videos/
│   └── dataset_links.txt
├── models/
│   ├── yolov8_weights.pt
│   ├── dwpose_model.pth
│   └── trained_model/
├── src/
│   ├── main.py
│   ├── gesture_recognition.py
│   ├── speech_to_text.py
│   ├── text_to_speech.py
│   ├── avatar_animation.py
│   └── utils.py
├── notebooks/
│   ├── data_preprocessing.ipynb
│   └── model_training.ipynb
└── docs/
    ├── architecture_diagram.png
    └── usage_guide.md


