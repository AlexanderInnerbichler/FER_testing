import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "dima806/facial_emotions_image_detection"

print("Loading model...")
extractor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()
print("Model ready. Press 'q' to quit.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (device 0)")

INFERENCE_EVERY_N = 3  # run model every N frames to keep display smooth
frame_count = 0
last_labels: list[tuple[str, float]] = []

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

    for entry in last_labels:
        label, confidence, x, y, w, h = entry
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
        text = f"{label} {confidence:.0%}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    cv2.imshow("Facial Emotion Detection", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
