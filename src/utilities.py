import cv2
import os
import streamlit as st
import tempfile
from transformers import pipeline

pipe = pipeline("image-to-text", model="ghanahmada/trocr-base-plate-number")

def save_uploaded(upload_file):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, upload_file.name)
    with open(file_path, "wb") as f:
        f.write(upload_file.getbuffer())
    return file_path

def process_video(video_path, model, confidence_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(tempfile.mkdtemp(), "output.mp4")
    out = None
    frame_count = 0
    processed_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 15 == 0:
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
                save_path = os.path.join(tempfile.mkdtemp(), f"frame_{frame_count}.jpg")
                cv2.imwrite(save_path, cropped_image)
                result_ocr = pipe(save_path)[0]
                processed_frames.append((annotated_frame, result_ocr['generated_text']))
                # col1, col2 = st.columns(2)
                # with col1:
                #     st.image(annotated_frame)
                # with col2:
                #     st.markdown(f"plate: {result_ocr['generated_text']}")
                
            if out is None:
                height, widht, _ = frame.shape
                out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (height, widht))
            out.write(annotated_frame)
            # st_frame.image(annotated_frame, channels="BGR")
        frame_count += 1

    cap.release()
    if out is not None:
        out.release()
    return out_path, processed_frames
