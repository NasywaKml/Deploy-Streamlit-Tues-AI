import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import gdown
import os
from ultralytics import YOLO


# Explicitly specify YOLOv5 local path
if not os.path.exists("best.pt"):
    model_download_id = "1EiG6pPnhVokLRBwUt9rNk6FhsvkUv1ri"
    gdown.download(f'https://drive.google.com/uc?id={model_download_id}', f'best.pt', quiet=False)


model = YOLO("best.pt")


# Streamlit UI
st.title("Deteksi Burung dengan YOLOv5")
st.sidebar.title("Pilihan Mode")

# Pilih mode operasi
mode = st.sidebar.radio("Pilih mode:", ["Live Webcam", "Upload Video", "Upload Gambar"])


# Fungsi pembantu untuk mendeteksi objek pada gambar
def detect_objects(img):
    results = model(img)  # YOLO detection
    detected_img = img.copy()

    # Ensure results exist
    if results and results[0].boxes is not None:
        for detection in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = detection.tolist()  # Convert tensor to list
            if conf > 0.5:  # Confidence threshold
                label = f"{model.names[int(cls)]}: {conf:.2f}"
                cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(detected_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return detected_img


if mode == "Live Webcam":
    st.write("Mode Webcam Langsung")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("Error: Webcam tidak ditemukan atau tidak dapat diakses.")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frame = detect_objects(frame)
            FRAME_WINDOW.image(detected_frame)

        cap.release()


elif mode == "Upload Video":
    st.write("Mode Upload Video")
    uploaded_video = st.file_uploader("Unggah file video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        temp_file = "temp_video.mp4"
        with open(temp_file, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file)
        FRAME_WINDOW = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frame = detect_objects(frame)
            FRAME_WINDOW.image(detected_frame)

        cap.release()


elif mode == "Upload Gambar":
    st.write("Silakan unggah gambar untuk deteksi objek.")

    uploaded_image = st.file_uploader("Unggah file gambar", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        try:
            img = Image.open(uploaded_image).convert("RGB")  # Load as RGB
            img_np = np.array(img)  # Convert to NumPy array

            detected_image = detect_objects(img_np)  # Object Detection

            # Show both original and detected images
            st.image([img, detected_image], caption=["Gambar Asli", "Hasil Deteksi"], use_column_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat mendeteksi gambar: {e}")
