import os
import argparse
import cv2
import numpy as np

VIT_MODEL_ID = "dima806/facial_emotions_image_detection"
STM32_MODEL_PATH = "stm32_model/fer2013_mobilenetv2_a035_128_float.keras"
THRESHOLD = 0.30
THUMB_SIZE = 100  # px — size of each emotion thumbnail in the gallery strip
SAVE_DIR = "captured_emotions"


def load_vit():
    """Big HuggingFace ViT (PyTorch). Returns (classify, class_names)."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    extractor = AutoImageProcessor.from_pretrained(VIT_MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(VIT_MODEL_ID)
    model.eval()
    class_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]

    def classify(face_bgr):
        pil_img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        inputs = extractor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        top_idx = int(probs.argmax())
        return class_names[top_idx], float(probs[top_idx])

    return classify, class_names


def load_stm32():
    """Newly trained MobileNetV2 a035 (TensorFlow). Returns (classify, class_names)."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    model = tf.keras.models.load_model(STM32_MODEL_PATH)

    def classify(face_bgr):
        rgb = cv2.cvtColor(cv2.resize(face_bgr, (128, 128)), cv2.COLOR_BGR2RGB)
        x = rgb.astype("float32") / 127.5 - 1.0
        probs = model(x[None, ...], training=False).numpy()[0]  # softmax already applied
        top_idx = int(probs.argmax())
        return class_names[top_idx], float(probs[top_idx])

    return classify, class_names


parser = argparse.ArgumentParser(description="Live facial emotion detection")
parser.add_argument("--model", choices=["vit", "stm32"], default="vit",
                    help="vit = big HuggingFace ViT (default); stm32 = newly trained MobileNetV2")
args = parser.parse_args()

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Loading {args.model} model...")
classify, ALL_EMOTIONS = load_vit() if args.model == "vit" else load_stm32()
captured: dict[str, np.ndarray] = {}  # emotion -> BGR face crop
captured_conf: dict[str, float] = {}  # emotion -> best confidence so far

print(f"Model ready ({args.model}). Capturing best sample per emotion above {THRESHOLD:.0%}. Press 'q' to quit.")

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


clear_button = (0, 0, 0, 0)  # x1, y1, x2, y2 — updated each frame for hit-testing


def clear_captured():
    captured.clear()
    captured_conf.clear()
    for name in os.listdir(SAVE_DIR):
        if name.endswith(".jpg"):
            os.remove(os.path.join(SAVE_DIR, name))
    print("Cleared captured emotions.")


def on_mouse(event, mx, my, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = clear_button
        if x1 <= mx <= x2 and y1 <= my <= y2:
            clear_captured()


WIN_NAME = f"Facial Emotion Detection [{args.model}]"
cv2.namedWindow(WIN_NAME)
cv2.setMouseCallback(WIN_NAME, on_mouse)

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
            label, confidence = classify(face_img)
            last_labels.append((label, confidence, x, y, w, h))

            if confidence >= THRESHOLD and confidence > captured_conf.get(label, 0):
                captured[label] = face_img.copy()
                captured_conf[label] = confidence
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

    bx2, by1 = display.shape[1] - 10, 10
    bx1, by2 = bx2 - 90, by1 + 26
    clear_button = (bx1, by1, bx2, by2)
    cv2.rectangle(display, (bx1, by1), (bx2, by2), (40, 40, 160), -1)
    cv2.rectangle(display, (bx1, by1), (bx2, by2), (80, 80, 220), 1)
    cv2.putText(display, "Clear", (bx1 + 14, by2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow(WIN_NAME, display)
    frame_count += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        clear_captured()

cap.release()
cv2.destroyAllWindows()
