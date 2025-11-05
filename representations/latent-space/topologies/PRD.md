# Product Requirements Document — Latent Topologies

## 1 Vision
Enable humans to *feel and interpret* latent space through a phone-native, offline experience.

## 2 Objectives
| Goal | Measure |
|------|----------|
| Embodied Exploration | Users report intuitive sense of relation |
| Interpretability | Users predict similar items > chance |
| Reflexivity | Annotations change local layout |
| Accessibility | < 5 s load, smooth > 50 fps |

## 3 User Stories
- Explore pre-bundled "constellation" of concepts.
- Add own text; app embeds locally and places it.
- Annotate clusters; export "latent walk".

## 4 Features (v1)
- Zoomable map (2D / 3D)
- Details drawer & neighbors
- Interpolation scrubber (text blend + audio glide)
- Sound + haptic feedback
- Offline Creator input and annotation

## 5 Constraints
- Dataset ≤ 10 k items (≈ 40 MB SQLite + coords)
- Model ≤ 100 MB quantized
- No network required

## 6 Deliverables
1. RN app bundle (Expo)
2. Open dataset + docs
3. Research paper / demo video

## 7 Timeline (~12 weeks)
| Phase | Focus | Duration |
|--------|-------|----------|
| M0 | Corpus + model prep | 2 wks |
| M1 | Map visualization | 2 wks |
| M2 | Audio + haptics | 2 wks |
| M3 | Creator mode | 3 wks |
| M4 | Annotations + export | 2 wks |
| M5 | Polish + study | 1 wk |
