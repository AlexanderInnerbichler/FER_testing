import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "dima806/facial_emotions_image_detection"

print("Loading model...")
extractor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()

# ViT expects (batch, channels, height, width) — 224x224 is the standard size
dummy = torch.zeros(1, 3, extractor.size["height"], extractor.size["width"])

# opset 17 is the minimum viable for ViT — LayerNormalization was introduced in opset 17
OUTPUT = "emotion_model_opset17.onnx"
print(f"Exporting to {OUTPUT}...")
torch.onnx.export(
    model, (dummy,), OUTPUT,
    input_names=["pixel_values"], output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
print(f"  -> {OUTPUT}")

print(f"Done. Labels: {list(model.config.id2label.values())}")
