import os
import pandas as pd
from pipeline import PolePipeline

YOLO_MODEL = os.getenv("YOLO_MODEL_PATH", "best2.pt")
SAM_CKPT   = os.getenv("SAM_CKPT_PATH", "sam_vit_h_4b8939.pth")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs_final_pipeline")
CSV_PATH   = os.getenv("CSV_PATH", "images_v1.csv")

def run_batch():
    pipeline = PolePipeline(
        yolo_model_path=YOLO_MODEL,
        sam_checkpoint=SAM_CKPT,
        output_folder=OUTPUT_DIR,
    )

    df = pd.read_csv(CSV_PATH)
    all_results = []

    for _, row in df.iterrows():
        image_path = row["Image_Path"]
        lat, lon = row["Lat"], row["Lon"]
        filename = os.path.basename(image_path)
        res = pipeline.process_image(image_path, filename, lat, lon)
        all_results.extend(res)

    out_df = pd.DataFrame(all_results)
    if "bbox" in out_df.columns:
        out_df = out_df.drop(columns=["bbox"], errors="ignore")
    os.makedirs("outputs", exist_ok=True)
    out_df.to_csv("outputs/pole_results_v1.csv", index=False)
    print("Saved: outputs/pole_results_v1.csv")

if __name__ == "__main__":
    run_batch()
