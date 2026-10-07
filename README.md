# Brain Tumor Classification with Explainable AI

**MRI image classification with CNN ensembles and Grad-CAM utilities.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)

An academic computer vision project covering four MRI image classes: **glioma, meningioma, no tumor, and pituitary tumor**. The supplied code contains a weighted ensemble, InceptionV3 training/evaluation, DenseNet explanation utilities, and two Flask demos.

## Pavan's documented contribution and project reports

**Pavan Kumar Goud Amikula** is a named contributor and coauthor in the supplied academic reports.

| Report | Context and authorship |
| --- | --- |
| [CNN architecture comparison](docs/reports/cnn-architecture-comparison.pdf) | Pavan implemented and trained **InceptionV3 and MobileNetV3**. Naga Sreerama Pradyumna Tata implemented EfficientNetB3 and DenseNet201; both jointly contributed preprocessing, evaluation, confusion matrix analysis, and writing. |
| [Ensemble learning and explainable AI study](docs/reports/ensemble-explainable-ai-study.pdf) | Coauthored by Yashwanth Hudumula and Pavan Kumar Goud Amikula; investigates ensemble performance, class-level reliability, and Grad-CAM. |
| [Major-project report](docs/reports/major-project-report.pdf) | Team report naming TNS Pradyumna, A Pavan Kumar Goud, H Yashwanth, and THS Chakradhar, supervised by Dr. M. Arathi. |

The central problem is that high aggregate accuracy can conceal class-specific errors. The work compares architectures and their efficiency/reliability trade-offs, then explores ensembles and visual explanations. Pavan's documented implementation role is specifically established by the CNN comparison report; the other reports establish team membership or coauthorship without a separate task breakdown.

The major-project title mentions segmentation, but its conclusion treats segmentation as future work. The repository's Grad-CAM utilities should not be presented as a completed, validated segmentation system.

**Results context:** the major-project and ensemble-study PDFs report 99.01% ensemble accuracy. The code archive's saved summary reports 96.1098% on 1,311 images. These different historical artifacts are preserved separately; the reports do not replace the repository's saved evaluation evidence or establish a fresh reproduction.

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

Published here as a team portfolio project from the supplied archive and accompanying reports. Its original archive README identifies **Yashwanth Hudumula** as author; that source attribution is preserved. The reports additionally document Pavan's team membership, coauthorship, and specific CNN-comparison contribution above. This does not claim sole authorship or grant a new software license. Medical image uploads, image-level prediction records, caches, and model binaries are excluded. This is an educational classification project; clinical use has not been validated.
