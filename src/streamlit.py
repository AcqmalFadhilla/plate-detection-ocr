import streamlit as st
import os

from ultralytics import YOLO
from utilities import save_uploaded, process_video

st.title("Video Plate Detection and OCR")
model = YOLO("results/models/best.pt")

data = st.file_uploader("Pilih file", type=["mp4", "mov", "avi", "mkv"])
dir_file = "/Users/acqmallatief/Project/plate-detection-ocr/results/visual"

if data is not None:
    video_path = save_uploaded(data)
    with st.spinner("Processing video..."):
        result_video, processed_frames =  process_video(video_path=video_path, model=model)

    print(result_video)
    st.write("Video processed. Displaying results:")
    st.video(result_video)

    for annotated_frame, ocr_text in processed_frames:
        col1, col2 = st.columns(2)
        with col1:
            st.image(annotated_frame, channels="BGR")
        with col2:
            st.markdown(f"Plate: {ocr_text}")


    


