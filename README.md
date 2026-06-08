# FER_testing — Facial Emotion Recognition

Facial emotion recognition, from a desktop webcam demo to a quantized model deployable on an
**STM32N657** (Neural-ART NPU). Seven emotions: `angry, disgust, fear, happy, neutral, sad, surprise`.

## Contents
- [`emotion_cam.py`](emotion_cam.py) — live webcam demo (PyTorch / HuggingFace ViT).
- [`stm32_model/`](stm32_model/) — the compact int8 model trained for the STM32N657, plus its config.
- [`export_onnx.py`](export_onnx.py) — exports the ViT demo model to ONNX (kept for reference).

---

## 1. Webcam demo

```bash
pip install -r requirements.txt
python emotion_cam.py                 # big ViT (default); press 'q' to quit
python emotion_cam.py --model stm32   # the newly trained MobileNetV2 (needs TensorFlow)
```

It detects faces (Haar cascade), classifies each, and keeps a gallery strip of the best capture per
emotion (captures at ≥30% confidence, overwriting when a higher-confidence sample appears). Saves
crops to `captured_emotions/`.

`--model` switches the classifier:
- **`vit`** (default) — the big `dima806/facial_emotions_image_detection` ViT (PyTorch).
- **`stm32`** — the int8-bound MobileNetV2 from [`stm32_model/`](stm32_model/) (TensorFlow). This is
  the model that runs on the board, so the demo shows what to expect on-device.

Each backbone's dependencies are imported lazily, so you only need PyTorch for `vit` or TensorFlow
for `stm32` (the `stm32ai` conda env from §2 has both).

> The ViT (~343 MB) is great on a PC but **cannot run on an STM32** — it's too large and its
> transformer ops map poorly to the Neural-ART NPU. That motivated training the model below.

---

## 2. STM32N657 model

A **MobileNetV2 (alpha 0.35, 128×128×3)** classifier — ST's canonical N6 backbone — trained with
[ST's `stm32ai-modelzoo-services`](https://github.com/STMicroelectronics/stm32ai-modelzoo-services).
No custom training code: the whole train → quantize → evaluate pipeline is driven by one YAML config.

### Results (FER2013 test set, 7,178 images)
| Model | Accuracy | Size |
|-------|----------|------|
| Float (Keras) | 65.58% | 2.2 MB |
| **Int8 (TFLite)** | **65.09%** | **613 KB** |

Quantization drop is only **0.49%**. ~65% is the normal band for FER2013 with a small model
(noisy dataset; human accuracy ≈ 65–70%). Artifacts live in [`stm32_model/`](stm32_model/).

### How it was trained

**Environment** — the modelzoo requires **Python 3.12.9 + TensorFlow 2.18** (system Python is
PyTorch-only), so a dedicated conda env was used:
```bash
# Miniforge installed at ~/miniforge3
conda create -n stm32ai python=3.12.9 -y
conda activate stm32ai
git clone https://github.com/STMicroelectronics/stm32ai-modelzoo-services
pip install -r stm32ai-modelzoo-services/requirements.txt
pip install "nvidia-cuda-nvcc-cu12==12.6.*"   # provides ptxas for TF's XLA JIT (else SIGABRT)
```

**Dataset** — FER2013 in the class-folder layout the framework expects:
```
datasets/fer2013/
  train/<emotion>/*.jpg   (28,709 images)
  test/<emotion>/*.jpg    ( 7,178 images)
```
Downloading 28k individual files from HuggingFace got rate-limited (HTTP 429), so the canonical
single-file `fer2013.csv` was used instead and decoded into folders (each row is a 48×48 grayscale
image; the `Usage` column gives the train/test split). See `stm32_model/` notes for the converter.

**Config** — `stm32_model/fer2013_chain_tqe.yaml` selects:
- `operation_mode: chain_tqe` (train → quantize → evaluate in one run)
- `model: mobilenetv2_a035`, `input_shape: (128,128,3)`, `pretrained: True` (ImageNet transfer learning)
- `dataset_name: custom_dataset`, `validation_split: 0.2`, `quantization_split: 0.3`
- preprocessing: rescale to [-1,1], `color_mode: rgb` (grayscale auto-expanded to 3 channels)
- training: Adam lr 1e-3, batch 64, up to 150 epochs, `EarlyStopping(patience=20)` + `ReduceLROnPlateau`
- quantization: `TFlite_converter`, PTQ, `uint8` input → int8 model

**Run** — a single command kicks off all three stages:
```bash
cd stm32ai-modelzoo-services/image_classification
python stm32ai_main.py --config-path ./config_file_examples/ --config-name fer2013_chain_tqe.yaml
```
1. **Train** — fine-tuned on the RTX 3060; early-stopped at **epoch 68/150** (~12 min) on a ~65%
   validation plateau, restoring best weights.
2. **Quantize** — post-training int8 quantization, calibrated on 30% of the training data.
3. **Evaluate** — scored float (65.58%) and int8 (65.09%) on the held-out test set.

### Deploy to STM32N6570-DK
Requires **ST Edge AI Core** (≥2.0) + **STM32CubeIDE** + **STM32CubeProgrammer**. Run the modelzoo
in `deployment` mode pointing `model_path` at `stm32_model/fer2013_mobilenetv2_a035_128_int8.tflite`,
board `STM32N6570-DK`, using the bundled `application_code/image_classification/STM32N6` app.

### Ideas to push past 65%
Larger alpha (`mobilenetv2_a050`/`a075`), class-imbalance handling (only 436 "disgust" samples),
or a cleaner dataset (RAF-DB / AffectNet) — none require pipeline changes, just the config.
