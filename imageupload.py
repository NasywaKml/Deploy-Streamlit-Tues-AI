import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
model = torch.hub.load('ultralytics/yolov5', 'custom', path= r"D:\Tubes AI\best.pt")


# Streamlit UI
st.title("Deteksi Burung dengan YOLOv5")
st.sidebar.title("Pilihan Mode")

# Pilih mode operasi
mode = st.sidebar.radio("Pilih mode:", ["Live Webcam", "Upload Video", "Upload Gambar"])

# Fungsi deteksi objek
def detect_objects(img):
    results = model(img)  # Deteksi objek menggunakan model YOLOv5
    detected_img = np.array(img.copy())  # Salin gambar untuk ditampilkan

    # Iterasi hasil deteksi
    for detection in results.xyxy[0]:  
        x1, y1, x2, y2, conf, cls = detection
        if conf > 0.5:  # Threshold confidence
            label = f"{model.names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(detected_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return detected_img

# Mode Upload Gambar
if mode == "Upload Gambar":
    st.write("Silakan unggah gambar untuk deteksi objek.")

    uploaded_image = st.file_uploader("Unggah file gambar", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        try:
            # Load image using PIL
            img = Image.open(uploaded_image).convert("RGB")  # Ensure it's in RGB format

            # Convert PIL Image to NumPy
            img_np = np.array(img)

            # Detect objects in the image
            detected_image = detect_objects(img_np)

            # Show both original and detected images
            st.image([img, detected_image], caption=["Gambar Asli", "Hasil Deteksi"], use_column_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat mendeteksi gambar: {e}")
elif mode == "Upload Video":
    st.write("Mode Upload Video")
    # Mengunggah file video
    uploaded_video = st.file_uploader("Unggah file video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        # Menyimpan file video sementara
        temp_file = "temp_video.mp4"
        with open(temp_file, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_file)
        FRAME_WINDOW = st.image([])

        # Menampilkan video yang diunggah dan mendeteksi objek
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frame = detect_objects(frame)
            FRAME_WINDOW.image(detected_frame)

        cap.release()
elif mode == "Live Webcam":
    st.write("Mode Webcam Langsung")
    FRAME_WINDOW = st.image([])

    # Membuka koneksi ke webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Webcam tidak ditemukan atau tidak dapat diakses.")
    else:
        # Menampilkan video dari webcam secara real-time
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frame = detect_objects(frame)
            FRAME_WINDOW.image(detected_frame)

    # Menutup koneksi webcam
    cap.release()

