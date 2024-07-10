import streamlit as st
from ultralytics import YOLO
from src.utilities import process_video, save_uploaded

st.title("Video Plate Detection and OCR")
model = YOLO("results/models/best.pt")

data = st.file_uploader("chose file", type=["mp4", "mov", "avi", "mkv"])
skip_frames = st.sidebar.slider("Skip frames", min_value=1, max_value=60, value=1, step=1, disabled=False)
print(skip_frames)

if data is not None:
    st.video(data)
    if st.button("detection"):
        video_path = save_uploaded(data)
        frame = st.empty()   
        process_video(video_path=video_path, 
                      st_frame=frame,
                      skip_frame=skip_frames,
                      model=model)
