import os
import pandas as pd
import streamlit as st
from pipeline import PolePipeline

st.set_page_config(page_title="Pole Identification & Lean Detection", layout="wide")
st.title("📸 Pole Identification & Lean Detection")

st.sidebar.header("Model Files")
yolo_model = st.sidebar.text_input("YOLO model path (mounted/file)", "best2.pt")
sam_ckpt = st.sidebar.text_input("SAM checkpoint path (mounted/file)", "sam_vit_h_4b8939.pth")
output_dir = st.sidebar.text_input("Output folder", "outputs_final_pipeline")

uploaded_csv = st.file_uploader("Upload CSV with columns: Image_Path, Lat, Lon", type=["csv"])

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    st.write("Sample rows:", df.head())

    if st.button("Run Pipeline"):
        pipeline = PolePipeline(yolo_model, sam_ckpt, output_folder=output_dir)
        results = []
        progress = st.progress(0)
        for i, row in df.iterrows():
            image_path = row["Image_Path"]
            lat, lon = row["Lat"], row["Lon"]
            filename = os.path.basename(image_path)
            res = pipeline.process_image(image_path, filename, lat, lon)
            results.extend(res)
            progress.progress(int((i+1) / len(df) * 100))
        st.success("Processing complete.")
        st.dataframe(pd.DataFrame(results))
        st.info("Find processed images under the outputs folder inside the container.")
else:
    st.info("Upload a CSV to start.")
