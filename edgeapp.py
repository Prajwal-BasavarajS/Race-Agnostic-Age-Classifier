import json
import time
import math
from dataclasses import dataclass
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# CONFIG


AGE_ORDER = ["0-12", "13-24", "25-39", "40-59", "60+"]

@dataclass
class EdgeConfig:
    camera_index: int = 0
    model_path: str = "models/efficientnetv2s_utk_age5_ftplus_final.pth"
    output_dir: str = "output"

    # inference
    device_preference: str = "auto"   # auto | mps | cpu
    img_size: int = 224
    confidence_threshold: float = 0.40

    # aggregation
    window_seconds: int = 20
    write_every_seconds: int = 2
    k_threshold: int = 1             # min count to publish bucket
    add_laplace_noise: bool = False
    dp_epsilon: float = 1.0

    # anti-overcounting
    cooldown_seconds: float = 3.0
    face_match_distance: float = 80.0

    # face detection
    min_face_size: tuple = (80, 80)
    scale_factor: float = 1.1
    min_neighbors: int = 5

    # UI
    show_window: bool = True
    draw_boxes: bool = True



# DEVICE

def get_device(pref: str = "auto") -> torch.device:
    if pref == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if pref == "cpu":
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



# MODEL

def build_model(num_classes: int = 5) -> nn.Module:
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_model(model_path: str, device: torch.device) -> nn.Module:
    model = build_model(num_classes=len(AGE_ORDER))
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model


def build_eval_tfms(img_size: int = 224):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(imagenet_mean, imagenet_std),
    ])


@torch.no_grad()
def predict_age_bucket(model: nn.Module, pil_img: Image.Image, tfms, device: torch.device):
    x = tfms(pil_img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    pred_idx = int(pred.item())
    return AGE_ORDER[pred_idx], float(conf.item())



# FACE DETECTION

def build_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade face detector.")
    return detector


def detect_faces(detector, frame_bgr, cfg: EdgeConfig):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=cfg.scale_factor,
        minNeighbors=cfg.min_neighbors,
        minSize=cfg.min_face_size,
    )
    return faces


# TRACKING / COOLDOWN


def center_of_box(box):
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


class CooldownTracker:
    def __init__(self, cooldown_seconds: float, max_distance: float):
        self.cooldown_seconds = cooldown_seconds
        self.max_distance = max_distance
        self.recent = deque(maxlen=200)  # each item: {"center": (cx,cy), "ts": timestamp}

    def should_count(self, box, now_ts: float) -> bool:
        c = center_of_box(box)

        for item in self.recent:
            if now_ts - item["ts"] <= self.cooldown_seconds:
                if euclidean(c, item["center"]) <= self.max_distance:
                    return False

        self.recent.append({"center": c, "ts": now_ts})
        return True


# PRIVACY-PRESERVING AGGREGATION

def laplace_noise(scale: float) -> float:
    return np.random.laplace(loc=0.0, scale=scale)


class Aggregator:
    def __init__(self, cfg: EdgeConfig):
        self.cfg = cfg
        self.window_start = time.time()
        self.window_counts = Counter()
        self.total_counts = Counter()
        self.last_write_ts = 0.0

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.live_json_path = out_dir / "live_counts.json"
        self.history_csv_path = out_dir / "history.csv"

        if not self.history_csv_path.exists():
            self.history_csv_path.write_text(
                "timestamp_iso,window_seconds,0-12,13-24,25-39,40-59,60+\n",
                encoding="utf-8"
            )

    def add(self, bucket: str):
        self.window_counts[bucket] += 1
        self.total_counts[bucket] += 1

    def _postprocess_counts(self, counts: Counter):
        processed = {}
        for bucket in AGE_ORDER:
            val = int(counts.get(bucket, 0))

            if self.cfg.add_laplace_noise:
                noisy = val + laplace_noise(scale=1.0 / max(self.cfg.dp_epsilon, 1e-6))
                val = max(0, int(round(noisy)))

            if val < self.cfg.k_threshold:
                val = 0

            processed[bucket] = val

        return processed

    def maybe_flush(self):
        now_ts = time.time()
        elapsed = now_ts - self.window_start

        if elapsed < self.cfg.window_seconds:
            if now_ts - self.last_write_ts >= self.cfg.write_every_seconds:
                self._write_live(partial=True)
                self.last_write_ts = now_ts
            return

        self._finalize_window()
        self.window_counts = Counter()
        self.window_start = now_ts
        self.last_write_ts = now_ts

    def _write_live(self, partial: bool):
        processed = self._postprocess_counts(self.window_counts)
        payload = {
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "window_seconds": self.cfg.window_seconds,
            "k_threshold": self.cfg.k_threshold,
            "partial_window": partial,
            "counts": processed,
            "total_counts": {b: int(self.total_counts.get(b, 0)) for b in AGE_ORDER},
        }
        self.live_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _finalize_window(self):
        processed = self._postprocess_counts(self.window_counts)

        payload = {
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "window_seconds": self.cfg.window_seconds,
            "k_threshold": self.cfg.k_threshold,
            "partial_window": False,
            "counts": processed,
            "total_counts": {b: int(self.total_counts.get(b, 0)) for b in AGE_ORDER},
        }
        self.live_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        row = [
            payload["timestamp_iso"],
            str(self.cfg.window_seconds),
            str(processed["0-12"]),
            str(processed["13-24"]),
            str(processed["25-39"]),
            str(processed["40-59"]),
            str(processed["60+"]),
        ]
        with self.history_csv_path.open("a", encoding="utf-8") as f:
            f.write(",".join(row) + "\n")


# MAIN APP

def crop_face(frame_bgr, box):
    x, y, w, h = box
    x = max(0, x)
    y = max(0, y)
    crop = frame_bgr[y:y+h, x:x+w]
    return crop


def bgr_to_pil(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def draw_overlay(frame, box, text):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.rectangle(frame, (x, y-25), (x+w, y), (0, 255, 0), -1)
    cv2.putText(
        frame,
        text,
        (x + 5, y - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )


def put_status(frame, text, line=1):
    y = 25 + (line - 1) * 25
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def main():
    cfg = EdgeConfig()
    device = get_device(cfg.device_preference)
    print("Using device:", device)

    model = load_model(cfg.model_path, device)
    tfms = build_eval_tfms(cfg.img_size)
    detector = build_face_detector()
    tracker = CooldownTracker(cfg.cooldown_seconds, cfg.face_match_distance)
    aggregator = Aggregator(cfg)

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Camera started. Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame.")
                break

            faces = detect_faces(detector, frame, cfg)
            now_ts = time.time()

            for box in faces:
                x, y, w, h = [int(v) for v in box]
                face_crop = crop_face(frame, (x, y, w, h))
                if face_crop.size == 0:
                    continue

                pil_face = bgr_to_pil(face_crop)
                bucket, conf = predict_age_bucket(model, pil_face, tfms, device)

                counted = False
                if conf >= cfg.confidence_threshold and tracker.should_count((x, y, w, h), now_ts):
                    aggregator.add(bucket)
                    counted = True

                if cfg.show_window and cfg.draw_boxes:
                    tag = f"{bucket} | {conf:.2f}"
                    if counted:
                        tag += " | counted"
                    draw_overlay(frame, (x, y, w, h), tag)

            aggregator.maybe_flush()

            if cfg.show_window:
                elapsed = int(time.time() - aggregator.window_start)
                remaining = max(0, cfg.window_seconds - elapsed)
                live_counts = aggregator.window_counts

                put_status(frame, f"Window remaining: {remaining}s", line=1)
                put_status(
                    frame,
                    f"Counts: 0-12={live_counts.get('0-12',0)}  13-24={live_counts.get('13-24',0)}  "
                    f"25-39={live_counts.get('25-39',0)}  40-59={live_counts.get('40-59',0)}  60+={live_counts.get('60+',0)}",
                    line=2
                )
                put_status(frame, "Privacy mode: only aggregated outputs are saved", line=3)

                cv2.imshow("Retail Age Analytics (Aggregated Only)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        aggregator._write_live(partial=True)
        print("Stopped cleanly.")


if __name__ == "__main__":
    main()