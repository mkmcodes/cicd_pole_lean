import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import shutil

# ---------- Haversine distance ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c  # distance in km

class PolePipeline:
    def __init__(self, yolo_model_path, sam_checkpoint, output_folder="outputs",
                 distance_threshold=0.05,  # km (50m default)
                 device="cuda" if torch.cuda.is_available() else "cpu"):

        # --- Models ---
        self.device = device
        self.yolo = YOLO(yolo_model_path)
        self.embed_threshold = 0.85  # Cosine similarity threshold for embeddings
        self.expand_ratio = 0.2  # bbox expand ratio for cropping
        # SAM setup
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam_predictor = SamPredictor(sam)

        # CNN (ResNet18) for embeddings
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()

        # Transform for CNN
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # --- Parameters ---
        self.distance_threshold = distance_threshold
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        # --- Database of processed poles ---
        # Format: {pole_id: {"lat": float, "lon": float, "embedding": np.array, "border_sums": dict}}
        self.poles_db = {}

        # Subfolders
        self.worked_folder = os.path.join(output_folder, "worked")
        self.failed_folder = os.path.join(output_folder, "failed")
        os.makedirs(self.worked_folder, exist_ok=True)
        os.makedirs(self.failed_folder, exist_ok=True)

        print(f"[INFO] Pipeline initialized. Saving results to {output_folder}")
        
    # ---------- Border sum helper ----------
    def _get_border_sums(self, image_rgb, bbox, margin=15):
        """
        Compute sum of RGB pixel values in a margin around bbox.
        bbox: [x1, y1, x2, y2]
        """
        h, w, _ = image_rgb.shape
        x1, y1, x2, y2 = bbox

        left  = image_rgb[y1:y2, max(0, x1 - margin):x1]
        right = image_rgb[y1:y2, x2:min(w, x2 + margin)]
        up    = image_rgb[max(0, y1 - margin):y1, x1:x2]
        down  = image_rgb[y2:min(h, y2 + margin), x1:x2]
        full  = image_rgb[max(0, y1 - margin):min(h, y2 + margin),
                          max(0, x1 - margin):min(w, x2 + margin)]

        sums = {
            "sum_left": int(left.sum()) if left.size > 0 else 0,
            "sum_right": int(right.sum()) if right.size > 0 else 0,
            "sum_up": int(up.sum()) if up.size > 0 else 0,
            "sum_down": int(down.sum()) if down.size > 0 else 0,
            "sum_full": int(full.sum()) if full.size > 0 else 0,
        }
        return sums

    # ---------- Pole ID Pipeline ----------
    def run_pole_id_pipeline(self, image_rgb, filename, bboxes):
        poles_info = []
        if not bboxes:
            print(f"[WARN] No YOLO detection in {filename}")
            return poles_info

        for bbox in bboxes:
            border_sums = self._get_border_sums(image_rgb, bbox, margin=15)
            pole_id = f"{border_sums['sum_left']}_{border_sums['sum_right']}_{border_sums['sum_up']}_{border_sums['sum_down']}_{border_sums['sum_full']}"
            
            self.poles_db[pole_id] = {"lat": None, "lon": None, "embedding": None, "border_sums": border_sums}

            poles_info.append({
                "filename": filename,
                "pole_id": pole_id,
                "bbox": bbox,
                "similarity": None,
                "status": "New pole (far location)",
                **border_sums
            })
        return poles_info

    # ---------- SAM segmentation ----------
    def _mask_pole_with_sam(self, image, bbox):
        self.sam_predictor.set_image(image)
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(bbox)[None, :],
            multimask_output=True,
        )
        best_mask = masks[scores.argmax()]
        pole_only = cv2.bitwise_and(image, image, mask=best_mask.astype("uint8") * 255)
        x1, y1, x2, y2 = bbox
        return pole_only[y1:y2, x1:x2]
    
    # ---------- CNN embedding ----------
    def _get_embedding(self, pole_img):
        if pole_img is None or pole_img.size == 0:
            return None
        img_tensor = self.transform(pole_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.cnn(img_tensor).squeeze()
        embedding = feat.cpu().numpy()
        return embedding / np.linalg.norm(embedding)

    # ---------- Compare embeddings ----------
    def _compare_embeddings(self, new_embedding, threshold=0.9):
        if not self.poles_db:
            print("[DEBUG] poles_db is empty", self.poles_db)
            return None, 0.0  # No DB yet
        
        db_items = [(k, v["embedding"]) for k, v in self.poles_db.items() if v["embedding"] is not None]
        if not db_items:
            print("[DEBUG] No embeddings in DB yet")
            return None, 0.0
        keys, db_embeddings = zip(*db_items)
        sims = cosine_similarity([new_embedding], list(db_embeddings))[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        matched_pole_id = keys[best_idx]
        return matched_pole_id, best_score

    # ---------- Embedding Pipeline ----------
    def run_embedding_pipeline(self, image_rgb, filename, bboxes, threshold=0.85):
        poles_info = []
        if not bboxes:
            print(f"[WARN] No YOLO detection in {filename}")
            return poles_info

        for bbox in bboxes:
            roi = self._mask_pole_with_sam(image_rgb, bbox)
            embedding = self._get_embedding(roi)
            if embedding is None:
                poles_info.extend(self.run_pole_id_pipeline(image_rgb, filename, [bbox]))
                continue

            matched_pole_id, score = self._compare_embeddings(embedding, threshold)

            if matched_pole_id and score >= threshold:
                border_sums = self._get_border_sums(image_rgb, bbox, margin=15)
                poles_info.append({
                    "filename": filename,
                    "pole_id": matched_pole_id,
                    "similarity": score,
                    "status": "Matched existing pole",
                    "bbox": bbox,
                    **border_sums
                })
            else:
                border_sums = self._get_border_sums(image_rgb, bbox, margin=15)
                pole_id = f"{border_sums['sum_left']}_{border_sums['sum_right']}_{border_sums['sum_up']}_{border_sums['sum_down']}_{border_sums['sum_full']}"
                
                self.poles_db[pole_id] = {
                    "lat": None,
                    "lon": None,
                    "embedding": embedding,
                    "border_sums": border_sums
                }
                
                poles_info.append({
                    "filename": filename,
                    "pole_id": pole_id,
                    "bbox": bbox,
                    "similarity": score if score > 0 else None,
                    "status": f"New pole (low similarity: {score:.2f})" if score > 0 else "New pole (no match)",
                    **border_sums
                })

        return poles_info

    def _save_combined(self, save_path, yolo_img, roi, edges, hough_img, angle, pole_id):
        fig, axs = plt.subplots(1, 4, figsize=(20, 8))
        axs[0].imshow(cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("YOLO Detection"); axs[0].axis("off")
        axs[1].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"ROI {pole_id}"); axs[1].axis("off")
        axs[2].imshow(edges, cmap="gray")
        axs[2].set_title("Canny Edges"); axs[2].axis("off")
        axs[3].imshow(cv2.cvtColor(hough_img, cv2.COLOR_BGR2RGB))
        axs[3].set_title(f"Hough - Angle {angle if angle is not None else 'N/A'}°")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path   

    # ---------------- LEAN DETECTION ----------------
    def run_lean_detection(self, image_rgb, poles_info, yolo_img):
        """
        Detects pole leaning angle using Canny + Hough transform.
        Stores angle per pole_id in poles_info.
        Also saves processed visualization and adds a clickable link column.
        """
        results = []
        for pole in poles_info:
            if "bbox" not in pole:
                continue

            x1, y1, x2, y2 = pole["bbox"]

            roi = self._mask_pole_with_sam(image_rgb, (x1, y1, x2, y2))

            if roi is None or roi.size == 0:
                pole["lean_angle_deg"] = None
                pole["processed_image_link"] = None
                results.append(pole)
                continue

            # Preprocess ROI
            h, w = roi.shape[:2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
            edges = cv2.Canny(morph, 50, 100)

            # Hough line detection
            min_len = max(20, int(h * 0.3))
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                    threshold=40,
                                    minLineLength=min_len,
                                    maxLineGap=100)

            angle_deg = None
            hough_img = np.zeros_like(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
            if lines is not None:
                best_line, best_len = None, 0
                for l in lines:
                    x1l, y1l, x2l, y2l = l[0]
                    ang = abs(math.degrees(math.atan2((y2l - y1l), (x2l - x1l))))
                    if 60 <= ang <= 120:  # near vertical
                        length = np.linalg.norm([x2l - x1l, y2l - y1l])
                        if length > best_len:
                            best_len = length
                            best_line = (x1l, y1l, x2l, y2l)

                if best_line is not None:
                    x1l, y1l, x2l, y2l = best_line
                    cv2.line(hough_img, (x1l, y1l), (x2l, y2l), (255, 255, 255), 3)

                    dx, dy = x2l - x1l, y2l - y1l
                    angle_rad = math.atan2(dx, dy)
                    angle_deg = math.degrees(angle_rad)

                    if angle_deg > 90:
                        angle_deg -= 180
                    elif angle_deg < -90:
                        angle_deg += 180

            pole["lean_angle_deg"] = angle_deg

            # -------- Save processed visualization --------
            save_name = f"{os.path.splitext(pole['filename'])[0]}_{pole['pole_id']}.jpg"
            save_path = os.path.join(self.worked_folder, save_name)

            processed_path = self._save_combined(
                save_path,
                yolo_img ,
                roi,
                edges,
                hough_img,
                angle_deg,
                pole["pole_id"]
            )

            # add clickable link
            pole["processed_image_link"] = f"file://{os.path.abspath(processed_path)}"

            results.append(pole)

        return results

    # ---------------- MAIN STEP per image (no CSV loop here) ----------------
    def process_image(self, image_path, filename, lat, lon):
        """
        Main pipeline:
        - YOLO detection → bboxes
        - Choose ID/Embedding pipeline
        - Lean Detection
        """
        results = self.yolo.predict(image_path, conf=0.6, verbose=False)
        if not results:
            shutil.copy(image_path, os.path.join(self.failed_folder, filename))
            return [] 
 
        r = results[0]
        yolo_img = r.orig_img.copy()
        
        has_detection = False
        detections = []
        if hasattr(r, "obb") and r.obb is not None and len(r.obb) > 0:
            detections = [("obb", ob) for ob in r.obb.xyxyxyxy]
            has_detection = True
        elif r.boxes is not None and len(r.boxes) > 0:
            detections = [("box", box) for box in r.boxes]
            has_detection = True

        if not has_detection:
            shutil.copy(image_path, os.path.join(self.failed_folder, filename))
            return []

        bboxes = []
        for det in detections:
            if det[0] == "obb":
                pts = det[1].cpu().numpy().astype(int).reshape((4, 2))
                cv2.polylines(yolo_img, [pts], True, (0, 255, 0), 2)
                # For simplicity, convert the rotated box to an axis-aligned bbox
                x1, y1 = pts[:,0].min(), pts[:,1].min()
                x2, y2 = pts[:,0].max(), pts[:,1].max()
                bbox = [int(x1), int(y1), int(x2), int(y2)]
            else:
                x1, y1b, x2, y2b = det[1].xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(yolo_img, (x1, y1b), (x2, y2b), (0, 255, 0), 2)
                bw, bh = (x2 - x1), (y2b - y1b)
                x1 = max(0, x1 - int(bw * self.expand_ratio))
                y1b = max(0, y1b - int(bh * self.expand_ratio))
                x2 = min(r.orig_img.shape[1] - 1, x2 + int(bw * self.expand_ratio))
                y2b = min(r.orig_img.shape[0] - 1, y2b + int(bh * self.expand_ratio))
                bbox = [x1, y1b, x2, y2b]
            bboxes.append(bbox)

        image_rgb = cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2RGB)

        if not self.poles_db:
            poles_info = self.run_embedding_pipeline(image_rgb, filename, bboxes, threshold=self.embed_threshold)
        else:
            distances = []
            for pole_id, info in self.poles_db.items():
                if info["lat"] is not None and info["lon"] is not None:
                    d = haversine(lat, lon, info["lat"], info["lon"])
                    distances.append(d)
            min_dist = min(distances) if distances else float("inf")
            if min_dist > self.distance_threshold:
                poles_info = self.run_pole_id_pipeline(image_rgb, filename, bboxes)
            else:
                poles_info = self.run_embedding_pipeline(image_rgb, filename, bboxes, threshold=self.embed_threshold)

        # Update lat/lon in DB for new poles
        for p in poles_info:
            if "pole_id" in p and p["pole_id"] in self.poles_db:
                self.poles_db[p["pole_id"]]["lat"] = lat
                self.poles_db[p["pole_id"]]["lon"] = lon

        # Lean detection visuals
        self.run_lean_detection(image_rgb=r.orig_img, poles_info=poles_info, yolo_img=yolo_img)
        return poles_info
