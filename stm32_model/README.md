# FER model for STM32N657 (Neural-ART NPU)

A compact facial-emotion classifier trained with [ST's stm32ai-modelzoo-services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services)
to replace the original ViT (which is far too large for an STM32 and maps poorly to the NPU).

## Model
- **Architecture:** MobileNetV2, alpha 0.35, input 128×128×3 (ST's canonical N6 image-classification backbone)
- **Classes (7):** `angry, disgust, fear, happy, neutral, sad, surprise`
- **Dataset:** FER2013 (28,709 train / 7,178 test)

## Results (FER2013 test set)
| Model | Accuracy | Size |
|-------|----------|------|
| Float (Keras) | 65.58% | 2.2 MB |
| **Int8 (TFLite)** | **65.09%** | **613 KB** |

Quantization drop is only 0.49%. ~65% is in the normal band for FER2013 with a small model
(the dataset is noisy; human accuracy ≈ 65–70%).

## Files
- `fer2013_mobilenetv2_a035_128_int8.tflite` — **deployment artifact** (int8 PTQ, uint8 input). Feed this to ST Edge AI Core for the N657.
- `fer2013_mobilenetv2_a035_128_float.keras` — float reference model.
- `fer2013_chain_tqe.yaml` — the modelzoo config used (train → quantize → evaluate).

## Reproduce
From `image_classification/` in stm32ai-modelzoo-services (Python 3.12.9 env, TF 2.18):
```
python stm32ai_main.py --config-path ./config_file_examples/ --config-name fer2013_chain_tqe.yaml
```

## Deploy (STM32N6570-DK)
Requires ST Edge AI Core (≥2.0) + STM32CubeIDE + STM32CubeProgrammer, then run the modelzoo
in `deployment` mode pointing `model_path` at the int8 tflite above.
