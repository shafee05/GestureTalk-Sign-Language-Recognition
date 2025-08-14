import streamlit as st
import cv2
import numpy as np
import os
import time
import threading
import queue
import speech_recognition as sr
import pyttsx3
from PIL import Image
from ultralytics import YOLO

# Configuration for video directory (modify this path as needed)
VIDEO_DIR = r"E:\sign\final\sign_videos"  # Path to generated videos
if not os.path.exists(VIDEO_DIR):
    st.error(f"Video directory {VIDEO_DIR} does not exist. Please ensure videos are generated.")
    st.stop()

# Load the YOLO model (previous model with mAP50 0.995)
model_path = r"C:\Users\Md.Shafee\runs\detect\train\weights\best.pt"
try:
    model = YOLO(model_path)
    class_names = model.names
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    st.stop()
    class_names = {0: 'bye', 1: 'hello', 2: 'no', 3: 'please', 4: 'sorry', 5: 'thankyou', 6: 'yes'}

# Expected UI classes (matches the model's classes)
ui_classes = ['bye', 'hello', 'no', 'please', 'sorry', 'thankyou', 'yes']

# Verify model classes match UI classes
if set(class_names.values()) != set(ui_classes):
    st.error("Model classes do not match expected UI classes. Please ensure the model is trained on the correct classes.")
    st.stop()

# Dictionary for local video files
sign_videos = {
    'bye': os.path.join(VIDEO_DIR, 'bye.mp4'),
    'hello': os.path.join(VIDEO_DIR, 'hello.mp4'),
    'no': os.path.join(VIDEO_DIR, 'no.mp4'),
    'please': os.path.join(VIDEO_DIR, 'please.mp4'),
    'sorry': os.path.join(VIDEO_DIR, 'sorry.mp4'),
    'thankyou': os.path.join(VIDEO_DIR, 'thankyou.mp4'),
    'yes': os.path.join(VIDEO_DIR, 'yes.mp4')
}

# Verify all video files exist
for sign, video_path in sign_videos.items():
    if not os.path.exists(video_path):
        st.error(f"Video file for '{sign}' not found at {video_path}. Please generate the video.")
        st.stop()

# Dictionary for sign meanings
sign_meanings = {
    'bye': 'A gesture to bid farewell or say goodbye.',
    'hello': 'A greeting gesture to say hi or welcome someone.',
    'no': 'A gesture to indicate negation or refusal.',
    'please': 'A polite gesture to make a request or ask for something.',
    'sorry': 'A gesture to express apology or regret.',
    'thankyou': 'A gesture to express gratitude or appreciation.',
    'yes': 'A gesture to indicate agreement or affirmation.'
}

# Dictionary for sign image URLs (sourced from Lifeprint ASL University)
sign_images = {
    'bye': 'https://www.lifeprint.com/asl101/signs/b/bye.jpg',
    'hello': 'https://www.lifeprint.com/asl101/signs/h/hello.jpg',
    'no': 'https://www.lifeprint.com/asl101/signs/n/no.jpg',
    'please': 'https://www.lifeprint.com/asl101/signs/p/please.jpg',
    'sorry': 'https://www.lifeprint.com/asl101/signs/s/sorry.jpg',
    'thankyou': 'https://www.lifeprint.com/asl101/signs/t/thankyou.jpg',
    'yes': 'https://www.lifeprint.com/asl101/signs/y/yes.jpg'
}

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Thread-safe queue for detected signs
sign_queue = queue.Queue()

# Webcam processing thread with YOLO detection
def webcam_thread():
    global frame, detected_signs, running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam. Please ensure a webcam is connected and try again.")
        st.session_state.webcam_active = False
        return
    last_detection_time = 0
    detection_interval = 2
    while running:
        ret, frm = cap.read()
        if not ret:
            st.error("Failed to capture video frame. Stopping webcam.")
            st.session_state.webcam_active = False
            break
        current_time = time.time()
        if current_time - last_detection_time >= detection_interval:
            try:
                results = model(frm, verbose=False)
                debug_detections = []
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            label = class_names[cls_id]
                            debug_detections.append(f"{label}: {conf:.2f}")
                            if conf > 0.3:  # Lowered threshold to capture more detections
                                if label in ui_classes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    cv2.rectangle(frm, (x1, y1), (x2, y2), (255, 255, 0), 2)
                                    cv2.putText(frm, f"{label} {conf:.2f}", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                                    if not detected_signs or detected_signs[-1] != label:
                                        detected_signs.append(label)
                                        sign_queue.put(label)  # Add to queue for sidebar
                                        last_detection_time = current_time
                # Debug output: show all detections (even below threshold)
                debug_text = "Detections: " + ", ".join(debug_detections) if debug_detections else "Detections: None"
                cv2.putText(frm, debug_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception as e:
                st.warning(f"Error during detection: {e}")
        frame = frm
        time.sleep(0.03)
    cap.release()

# Function for speech recognition
def recognize_speech():
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio).lower()
            st.write(f"Recognized: {text}")
            return text
    except sr.UnknownValueError:
        st.write("Could not understand audio. Please try again or use text input.")
        return None
    except sr.RequestError:
        st.write("Speech recognition API unavailable. Please use text input instead.")
        return None
    except AttributeError as e:
        st.write("PyAudio is not installed or configured correctly. Please use text input instead.")
        return None
    except Exception as e:
        st.write(f"Error during speech recognition: {e}. Please use text input instead.")
        return None

# Custom CSS for dark theme
st.markdown("""
    <style>
    body {
        background-color: #1E2526;
        color: white;
    }
    .stApp {
        background-color: #1E2526;
        color: white;
    }
    .stButton>button {
        background-color: #2E3B3E;
        color: white;
        border: 1px solid #FFFFFF;
        border-radius: 5px;
    }
    .stTextArea textarea, .stTextInput input {
        background-color: #2E3B3E;
        color: white;
        border: 1px solid #FFFFFF;
    }
    .stRadio label {
        color: white;
    }
    .stSidebar {
        background-color: #2E3B3E;
        color: white;
    }
    .instructions-box {
        background-color: #1E2526;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #FFFFFF;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Main UI
st.title("Sign Language Translator")

# Sidebar for sign information
st.sidebar.header("Sign Information")
sign_info_placeholder = st.sidebar.empty()

# Instructions for new users in the sidebar
st.sidebar.markdown("""
<div class="instructions-box">
<h4>User Instructions</h4>
<p><b>Sign to Text / Speech:</b></p>
<ul>
    <li>Click "SIGN TO TEXT / SPEECH".</li>
    <li>Click "Toggle Webcam" and show a sign (e.g., "hello").</li>
    <li>Signs appear in "Transcribed Signs".</li>
    <li>Click "Convert to Speech" to hear the signs.</li>
</ul>
<p><b>Speech to Sign:</b></p>
<ul>
    <li>Click "SPEECH TO SIGN".</li>
    <li>Choose "Text" (type words) or "Speech" (click "Record Speech" and speak).</li>
    <li>Videos of signs will play.</li>
</ul>
<p><b>Sidebar:</b> Shows the latest sign's name, image, and meaning. "Unknown" means no sign detected.</p>
<p><b>Supported Signs:</b> bye, hello, no, please, sorry, thankyou, yes.</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'detected_signs' not in st.session_state:
    st.session_state.detected_signs = []
if 'frame' not in st.session_state:
    st.session_state.frame = None
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'last_detected_sign' not in st.session_state:
    st.session_state.last_detected_sign = None

# Global variables for webcam thread
frame = None
detected_signs = st.session_state.detected_signs
running = False

# Two main buttons for navigation
col1, col2 = st.columns(2)
with col1:
    if st.button("SIGN TO TEXT / SPEECH"):
        st.session_state.mode = "sign_to_text_speech"
        st.session_state.detected_signs = []
        st.session_state.last_detected_sign = None
with col2:
    if st.button("SPEECH TO SIGN"):
        st.session_state.mode = "speech_to_sign"
        st.session_state.detected_signs = []
        st.session_state.last_detected_sign = None

# SIGN TO TEXT / SPEECH Interface
if st.session_state.mode == "sign_to_text_speech":
    st.header("Sign to Text / Speech")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        webcam_placeholder = st.empty()
    with col2:
        if st.button("Toggle Webcam"):
            st.session_state.webcam_active = not st.session_state.webcam_active
            if st.session_state.webcam_active:
                running = True
                threading.Thread(target=webcam_thread, daemon=True).start()
            else:
                running = False
                webcam_placeholder.empty()
                st.session_state.last_detected_sign = None
                # Clear the sign queue when stopping
                while not sign_queue.empty():
                    sign_queue.get()
    
    # Process detected signs from the queue
    while not sign_queue.empty():
        label = sign_queue.get()
        if not st.session_state.detected_signs or st.session_state.detected_signs[-1] != label:
            st.session_state.detected_signs.append(label)
            st.session_state.last_detected_sign = label

    # Update sidebar with the latest detected sign
    if st.session_state.last_detected_sign and st.session_state.last_detected_sign in sign_meanings:
        sign = st.session_state.last_detected_sign
        with sign_info_placeholder.container():
            st.write(f"**Sign:** {sign.capitalize()}")
            try:
                st.image(sign_images[sign], caption=f"ASL gesture for '{sign}'", use_container_width=True)
            except Exception as e:
                st.write(f"Failed to load image for '{sign}'.")
            st.write(f"**Meaning:** {sign_meanings[sign]}")
    else:
        with sign_info_placeholder.container():
            st.write("**Sign:** Unknown")
            st.write("**Meaning:** No sign detected or unrecognized sign.")

    # Display webcam feed
    if st.session_state.webcam_active:
        while st.session_state.webcam_active and running:
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                webcam_placeholder.image(buffer.tobytes(), channels="BGR", use_container_width=True)
            time.sleep(0.03)
    else:
        webcam_placeholder.write("Webcam is off")

    sentence = " ".join(st.session_state.detected_signs)
    st.text_area("Transcribed Signs", value=sentence, height=100, disabled=True)

    if st.button("Convert to Speech") and sentence:
        try:
            engine.say(sentence)
            engine.runAndWait()
            st.write("Text converted to speech")
        except Exception as e:
            st.write(f"Error converting to speech: {e}")

# SPEECH TO SIGN Interface
if st.session_state.mode == "speech_to_sign":
    st.header("Speech to Sign")
    
    input_method = st.radio("Input Method", ("Text", "Speech"))
    
    if input_method == "Text":
        text_input = st.text_input("Enter text (e.g., hello please thankyou)")
    else:
        if st.button("Record Speech"):
            text_input = recognize_speech()
        else:
            text_input = None

    if text_input:
        words = text_input.lower().split()
        valid_words = [word for word in words if word in sign_videos]
        
        if valid_words:
            st.write("Displaying signs for:", " ".join(valid_words))
            # Update sidebar with the latest word being displayed
            st.session_state.last_detected_sign = valid_words[-1] if valid_words else None
            for word in valid_words:
                try:
                    # Display local video file
                    with open(sign_videos[word], "rb") as video_file:
                        st.video(video_file)
                except Exception as e:
                    st.image(Image.new('RGB', (640, 360), color='gray'), 
                             caption=f"Placeholder for {word} (Video failed to load: {e})")
        else:
            st.write("No valid signs found in input. Supported signs:", ", ".join(sign_videos.keys()))
            st.session_state.last_detected_sign = None

    # Update sidebar in Speech to Sign mode
    if st.session_state.last_detected_sign and st.session_state.last_detected_sign in sign_meanings:
        sign = st.session_state.last_detected_sign
        with sign_info_placeholder.container():
            st.write(f"**Sign:** {sign.capitalize()}")
            try:
                st.image(sign_images[sign], caption=f"ASL gesture for '{sign}'", use_container_width=True)
            except Exception as e:
                st.write(f"Failed to load image for '{sign}'.")
            st.write(f"**Meaning:** {sign_meanings[sign]}")
    else:
        with sign_info_placeholder.container():
            st.write("**Sign:** Unknown")
            st.write("**Meaning:** No sign detected or unrecognized sign.")

if __name__ == "__main__":
    pass