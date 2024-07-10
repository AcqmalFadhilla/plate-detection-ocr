import streamlit as st
from ultralytics import YOLO
from src.utilities import process_image, save_uploaded

st.title("Image Plate Detection and OCR")
model = YOLO("results/models/best.pt")

data = st.file_uploader("chose file", type=["jpg", "png"])

if data is not None:
    st.image(data)
    if st.button("detection"):
        image_path = save_uploaded(data)
        with st.spinner("In progress..."):
            process_image(data_path=image_path, 
                        model=model,
                        confidence_threshold=0.4)
