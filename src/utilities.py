import cv2
import os
import streamlit as st
import tempfile
from ultralytics import YOLO
from transformers import pipeline

def save_uploaded(upload_file):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, upload_file.name)
    with open(file_path, "wb") as f:
        f.write(upload_file.getbuffer())
    return file_path

def save_image(model_pipe, image, name_file="tes"):
    save_path = os.path.join(tempfile.mkdtemp(), f"frame_{name_file}.jpg")
    cv2.imwrite(save_path, image)
    return model_pipe(save_path)[0]
    
def process_video(video_path, model, st_frame, skip_frame, confidence_threshold=0.5):
    pipe = pipeline("image-to-text", model="ghanahmada/trocr-base-plate-number")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(tempfile.mkdtemp(), "output.mp4")
    out = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % skip_frame == 0:
            results = model(frame)
            annotated_frame = frame.copy()
            save_frame = False
            
            for box in results[0].boxes:
                if box.conf >= confidence_threshold:
                    save_frame = True
                    x_min, y_min, x_max, y_max = box.xyxy[0]
                    x_min = int(x_min)
                    y_min = int(y_min)
                    x_max = int(x_max)
                    y_max = int(y_max)
                    annotated_frame = results[0].plot()
                    cropped_image = annotated_frame[y_min:y_max, x_min:x_max]
                    break

            if save_frame:
                result_ocr = save_image(pipe, cropped_image, frame_count)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(annotated_frame)
                with col2:
                    st.markdown(f"plate: {result_ocr['generated_text']}")
                
            if out is None:
                height, widht, _ = frame.shape
                out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (height, widht))
            out.write(annotated_frame)
            st_frame.image(annotated_frame, channels="BGR")
        frame_count += 1

    cap.release()
    out.release()

def process_image(data_path, model, confidence_threshold):
    pipe = pipeline("image-to-text", model="ghanahmada/trocr-base-plate-number")
    image = cv2.imread(data_path)
    if image is None:
        raise ValueError("Gambar tidak dapat dibaca. Pastikan path benar.")
    cropped_image = None
    results = model(image)
    
    for box in results[0].boxes:
        if box.conf >= confidence_threshold:
            x_min, y_min, x_max, y_max = box.xyxy[0]
            x_min = int(x_min)
            y_min = int(y_min)
            x_max = int(x_max)
            y_max = int(y_max)
            annotated_frame = results[0].plot()
            cropped_image = annotated_frame[y_min:y_max, x_min:x_max]
            break

    if cropped_image is not None:        
        result_ocr = save_image(pipe, cropped_image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(annotated_frame)
        with col2:
            st.markdown(f"plate: {result_ocr['generated_text']}")
    else:
        st.markdown("No plate detected with sufficient confidence.")


    

