# Tana

Flutter app for saree recognition and Indian textile heritage exploration. Classification runs fully on-device — no server or internet connection required for inference.

## Features

- **AI Lens** (`lib/screens/ai_lens_screen.dart`) — point the camera at a saree and classify it as Baluchari, Maheshwari, Negamam, or Phulkari
- **Onboarding** (`lib/screens/onboarding_screen.dart`) — introduction flow
- Custom theme in `lib/theme/tana_theme.dart`

## On-device classification

`lib/services/classifier_service.dart` runs a weighted two-model ensemble with `tflite_flutter`:

| Model | Asset | Ensemble weight |
|---|---|---|
| YOLO11m (classification) | `assets/models/yolo11m_4class.tflite` | 0.85 |
| ResNet50 | `assets/models/resnet50_4saree.tflite` | 0.15 |

The weights come from the ensemble verification in `classification/training/ensemble_compare.py`. Models are exported by `classification/training/export_to_tflite.py`.

## Running

```bash
flutter pub get
flutter run
```
