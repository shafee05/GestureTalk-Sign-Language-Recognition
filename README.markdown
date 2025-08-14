# Sign Language Translator Project - Requirements

This project requires the following libraries and packages to run the sign language translation application, video generation, and real-time detection scripts.

## System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.8 or higher
- **Hardware**:
  - Webcam for real-time sign detection
  - Microphone for speech-to-sign functionality (optional)
  - Sufficient disk space for dataset and generated videos (~2-5 GB recommended)

## Python Libraries and Packages
Install the following packages using `pip`:

```bash
pip install streamlit opencv-python-headless numpy torch diffusers ultralytics speechrecognition pyttsx3 pillow
```

### Detailed Package List
1. **streamlit**: Web framework for the user interface.
   - Version: >= 1.20.0
   - Purpose: Runs the main application (`app1.py`) for sign-to-text/speech and speech-to-sign functionality.
2. **opencv-python-headless**: OpenCV library for webcam capture and image processing.
   - Version: >= 4.5.0
   - Purpose: Handles webcam feed and draws bounding boxes for detected signs.
3. **numpy**: Numerical computing library.
   - Version: >= 1.20.0
   - Purpose: Array manipulation for image processing and model inputs.
4. **torch**: PyTorch library for deep learning.
   - Version: >= 1.9.0
   - Purpose: Runs the Text-to-Video pipeline for video generation and supports YOLO model inference.
5. **diffusers**: Hugging Face library for generative models.
   - Version: >= 0.10.0
   - Purpose: Generates sign videos using the `cerspense/zeroscope_v2_576w` model in `generate_sign_videos.py`.
6. **ultralytics**: YOLOv8 implementation for object detection.
   - Version: >= 8.0.0
   - Purpose: Loads and runs the YOLO model for sign detection in `app1.py` and `real.py`.
7. **speechrecognition**: Library for speech recognition.
   - Version: >= 3.8.1
   - Purpose: Converts spoken input to text in the speech-to-sign mode.
8. **pyttsx3**: Text-to-speech library.
   - Version: >= 2.90
   - Purpose: Converts detected signs to spoken output in sign-to-text/speech mode.
9. **pillow**: Python Imaging Library (PIL) for image handling.
   - Version: >= 9.0.0
   - Purpose: Handles placeholder images in case of video loading failures.

### Additional Dependencies
- **PyAudio**: Required for `speechrecognition` to access the microphone.
  - Install with:
    ```bash
    pip install pyaudio
    ```
  - Note: On Windows, you may need to install PyAudio separately if you encounter issues. On Ubuntu, install `portaudio19-dev` first:
    ```bash
    sudo apt-get install portaudio19-dev
    ```

### Model Requirements
- **YOLOv8 Pre-trained Model**:
  - Path: `C:\Users\Md.Shafee\runs\detect\train\weights\best.pt` (as specified in `app1.py`)
  - Alternatively, train your own model using the dataset specified in `data.yaml` with the YOLOv8 command:
    ```bash
    yolo train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
    ```
- **Text-to-Video Model**:
  - Model: `cerspense/zeroscope_v2_576w` from Hugging Face
  - Automatically downloaded by the `diffusers` library when running `generate_sign_videos.py`.

### Dataset Requirements
- **American Sign Language Dataset**:
  - Source: Roboflow (https://universe.roboflow.com/project-et9mt/signlang-q5xad/dataset/1)
  - Format: YOLOv8
  - Structure: As specified in `data.yaml` with training, validation, and test images.
  - Classes: `bye`, `hello`, `no`, `please`, `sorry`, `thankyou`, `yes`

### Installation Steps
1. Install Python 3.8 or higher from https://www.python.org/.
2. Create and activate a virtual environment (optional):
   ```bash
   python -m venv sign_lang_env
   source sign_lang_env/bin/activate  # On Windows: sign_lang_env\Scripts\activate
   ```
3. Install all required packages:
   ```bash
   pip install streamlit opencv-python-headless numpy torch diffusers ultralytics speechrecognition pyttsx3 pillow pyaudio
   ```
4. Verify the installation by running:
   ```bash
   python -c "import streamlit, cv2, numpy, torch, diffusers, ultralytics, speech_recognition, pyttsx3, PIL"
   ```
   If no errors occur, the environment is set up correctly.

### Notes
- Ensure the paths in `app1.py` (e.g., `VIDEO_DIR`, `model_path`) match your local file system.
- The `diffusers` library may download large model files (~2-3 GB) the first time `generate_sign_videos.py` is run.
- For GPU acceleration (optional), install `torch` with CUDA support:
  ```bash
  pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
  ```
  Then modify `generate_sign_videos.py` to use `.to("cuda")` instead of `.to("cpu")`.

For further assistance, refer to the execution steps in `execution_steps.txt`.