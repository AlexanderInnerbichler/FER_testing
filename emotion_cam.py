import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "dima806/facial_emotions_image_detection"
THRESHOLD = 0.85
THUMB_SIZE = 100  # px — size of each emotion thumbnail in the gallery strip
SAVE_DIR = "captured_emotions"

os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading model...")
extractor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()

ALL_EMOTIONS: list[str] = [model.config.id2label[i] for i in range(len(model.config.id2label))]
captured: dict[str, np.ndarray] = {}  # emotion -> BGR face crop (THUMB_SIZE x THUMB_SIZE)

print(f"Model ready. Capturing one sample per emotion above {THRESHOLD:.0%}. Press 'q' to quit.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (device 0)")

INFERENCE_EVERY_N = 3
frame_count = 0
last_labels: list[tuple[str, float, int, int, int, int]] = []


def build_gallery(cam_width: int) -> np.ndarray:
    """Return a horizontal strip showing all emotions; captured ones show the face crop."""
    n = len(ALL_EMOTIONS)
    strip = np.zeros((THUMB_SIZE + 20, cam_width, 3), dtype=np.uint8)
    cell_w = cam_width // n

    for i, emotion in enumerate(ALL_EMOTIONS):
        x0 = i * cell_w
        if emotion in captured:
            thumb = cv2.resize(captured[emotion], (cell_w - 4, THUMB_SIZE - 4))
            strip[2:THUMB_SIZE - 2, x0 + 2:x0 + cell_w - 2] = thumb
            cv2.rectangle(strip, (x0, 0), (x0 + cell_w - 1, THUMB_SIZE - 1), (0, 200, 0), 2)
        else:
            cv2.rectangle(strip, (x0, 0), (x0 + cell_w - 1, THUMB_SIZE - 1), (60, 60, 60), 1)
            cv2.putText(strip, "?", (x0 + cell_w // 2 - 8, THUMB_SIZE // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)

        label_y = THUMB_SIZE + 14
        font_scale = 0.38
        text_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        text_x = x0 + (cell_w - text_size[0]) // 2
        color = (0, 220, 0) if emotion in captured else (130, 130, 130)
        cv2.putText(strip, emotion, (text_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    return strip


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if frame_count % INFERENCE_EVERY_N == 0:
        last_labels = []
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            inputs = extractor(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            top_idx = probs.argmax().item()
            label = model.config.id2label[top_idx]
            confidence = probs[top_idx].item()
            last_labels.append((label, confidence, x, y, w, h))

            if confidence >= THRESHOLD and label not in captured:
                captured[label] = face_img.copy()
                path = os.path.join(SAVE_DIR, f"{label}.jpg")
                cv2.imwrite(path, face_img)
                print(f"Saved {label} ({confidence:.1%}) -> {path}")

    for label, confidence, x, y, w, h in last_labels:
        color = (0, 200, 0) if confidence >= THRESHOLD else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        text = f"{label} {confidence:.0%}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    gallery = build_gallery(frame.shape[1])
    display = np.vstack([frame, gallery])

    done = len(captured)
    total = len(ALL_EMOTIONS)
    cv2.putText(display, f"Captured: {done}/{total}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    cv2.imshow("Facial Emotion Detection", display)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
