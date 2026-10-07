# Brain Tumor Classification with Explainable AI

**MRI image classification with CNN ensembles and Grad-CAM utilities.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)

An academic computer vision project covering four MRI image classes: **glioma, meningioma, no tumor, and pituitary tumor**. The supplied code contains a weighted ensemble, InceptionV3 training/evaluation, DenseNet explanation utilities, and two Flask demos.

## Architecture

MRI image → architecture-specific preprocessing → DenseNet201 / EfficientNetB3 / InceptionV3 → weighted probabilities → class prediction.

The ensemble script uses weights **0.5 DenseNet201, 0.3 EfficientNetB3, 0.2 InceptionV3**. Grad-CAM highlights regions influencing a prediction; it is not a tumor segmentation ground truth. The primary Flask demo uses a single DenseNet model, rather than the full ensemble.

## Saved evaluation evidence

The included `ensemble_outputs_modified_2/ensemble_summary.txt` reports:

| Metric | Saved result |
| --- | ---: |
| Test images | 1,311 |
| Accuracy | 96.11% |
| Macro precision | 0.9605 |
| Macro recall | 0.9585 |
| Macro F1 | 0.9589 |

These are historical saved results, not a new reproduction. The original archive README claims a different accuracy; this repository uses the included evaluation summary as its evidence.

## Repository guide

- `ensemble model.py`: ensemble loading, inference, and evaluation.
- `inception/`: InceptionV3 training, evaluation, single-image inference, and saved aggregate reports.
- `densenet/`: Grad-CAM and occlusion explanation scripts.
- `flask/`: single-model classification demo with Grad-CAM.
- `flask2/`: alternate classification demo; Grad-CAM is disabled in its route.
- `SOURCE_README.md`: original archive documentation and author credit, retained as provenance.

## Local setup

```bash
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install -r requirements.txt
python inception/train_inceptionv3.py --help
```

The root requirements list the imported libraries; it is not a locked, verified environment. Acquire appropriately licensed MRI data and compatible trained weights separately. The archive did not include the three `.h5` files expected by the ensemble. Place them in `saved models/`: `densenet201_final.h5`, `best_model.h5`, and `brain_tumor_inceptionv3.h5`. Create `Testing/` with the four class subdirectories before running `python "ensemble model.py"`.

For the primary local demo, set `BRAIN_TUMOR_MODEL` to a compatible DenseNet201 `.h5` path, run `python app.py` from `flask/`, and visit `http://127.0.0.1:5000`. Use `BRAIN_TUMOR_ROOT` to override the ensemble project root and `BRAIN_TUMOR_DENSENET_ROOT` for the DenseNet explanation folder. Utility scripts require matching TensorFlow/Keras serialization versions; rebuilding with skipped weights must be followed by independent evaluation.

## Attribution and scope

Published here as a portfolio project from the supplied archive. Its original README identifies **Yashwanth Hudumula** as author; that attribution is preserved. This publication does not establish sole authorship or grant a new software license. Medical image uploads, image-level prediction records, caches, and model binaries are excluded. This is an educational classification project; clinical use has not been validated.
