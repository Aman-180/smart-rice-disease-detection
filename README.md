# Rice Disease Detection System

An AI-powered system that helps rice farmers identify crop diseases from a simple leaf photo and get treatment advice — built for Tier 2/3 Indian farmers who often lack access to agricultural experts.

## Problem

Farmers frequently misidentify rice diseases from visual symptoms alone. A wrong diagnosis leads to the wrong chemical treatment being applied — wasting money, damaging the crop further, and reducing yield.

## Solution

A farmer uploads a photo of a diseased rice leaf. The system automatically detects the diseased region, classifies the disease, and returns treatment advice — no expert required.

## Technical Approach: Two-Stage Pipeline

Real farmer photos are rarely lab-quality — they're often blurry, distant, poorly lit, or cluttered with background noise. A single-stage classifier looking at the whole image tends to miss small diseased regions in these conditions.

To solve this, the system uses two models in sequence:

1. **YOLOv8** — detects and localizes the diseased region within the photo, cropping out irrelevant background (soil, other leaves, shadows).
2. **ResNet-50** — classifies the cropped region into one of five disease categories and returns matching treatment advice.

This mirrors how a human expert would examine a leaf: first isolate the affected area, then diagnose it.

## Classes

- `bacterialblight`
- `blast`
- `brownspot`
- `healthy`
- `tungro`

## Results

### Classification (ResNet-50)

Trained via transfer learning on ~7,500 images (5,245 train / 1,121 val / 1,125 test), 10 epochs on GPU.

**Overall test accuracy: 90.22%**

| Class            | Accuracy |
|-------------------|----------|
| bacterialblight    | 96.69%   |
| blast              | 98.15%   |
| brownspot          | 68.37%   |
| healthy            | 92.11%   |
| tungro             | 100.00%  |

**Known issue:** `brownspot` is frequently confused with `bacterialblight`, likely due to visually similar samples or label noise in the source dataset. This is flagged for further data cleaning rather than masked with inflated overall numbers.

### Detection (YOLOv8s)

Trained on ~600 bounding-box-labeled images (426 train / 121 valid / 62 test after an augmented 70/20/10 split), single class (`disease`).

**Final validation metrics:**

| Metric | Score |
|--------|-------|
| Precision | 0.606 |
| Recall | 0.500 |
| mAP50 | 0.538 |
| mAP50-95 | 0.195 |

Trained locally on an RTX 3050 (6GB VRAM) using `yolov8s.pt` as a pretrained checkpoint, with early stopping and augmentation (brightness, blur, rotation, noise) tuned to simulate real field-photo conditions.

## Project Structure

```
rice-disease-detection/
├── src/
│   ├── train_classification.py   # ResNet-50 training script
│   ├── evaluate_classification.py # Classifier evaluation
│   ├── train_yolo.py              # YOLOv8 training script
│   ├── inference.py               # Single-image disease classification
│   └── pipeline.py                # Combined YOLO + ResNet pipeline
├── dataset_yolo/                  # YOLO-format dataset (not tracked — see below)
├── models/
│   └── classification/
│       └── resnet50_rice.pth      # Trained classifier weights (see Model Weights below)
├── requirements.txt
└── README.md
```

## Build Challenges & Technical Obstacles

**1. Detecting disease in low-quality, real-world farmer photos**
Standard single-stage classifiers get distracted by background noise in blurry, poorly-lit field photos and can miss small diseased regions. Solved by splitting the problem into detection (YOLO) + classification (ResNet), so only the relevant cropped region reaches the classifier.

**2. Handling a large dataset and creating bounding box annotations**
Organizing ~7,500 images across five classes for classification, plus manually annotating bounding boxes on hundreds of images in Roboflow for detection, was a significant and detail-heavy effort — each box needed to tightly capture the diseased area without excess healthy leaf or background.

**3. YOLO training repeatedly crashing mid-checkpoint-save**
Training was interrupted multiple times by file I/O errors while saving checkpoints, traced to background processes (antivirus real-time scanning) locking the file mid-write. Resolved by excluding the project folder from real-time scanning and using YOLO's checkpoint-resume functionality to continue training without losing prior progress.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Usage

**Classify a single image:**
```bash
python src/inference.py --image path/to/leaf.jpg
```

**Run the full pipeline (detect + classify):**
```bash
python src/pipeline.py --image path/to/leaf.jpg
```

## Model Weights

Trained model weights (`resnet50_rice.pth`, YOLO `best.pt`) are not included in this repository due to file size. Available on request, or will be added via Google Drive / Hugging Face link.

## Honest Project Status

Both models — the YOLOv8 detector and the ResNet-50 classifier — are trained and validated independently. They have been connected into a working end-to-end pipeline (`src/pipeline.py`): an image goes in, YOLO detects and crops the diseased region, ResNet classifies it, and treatment advice is returned.

## Roadmap

- [ ] Clean up `brownspot` / `bacterialblight` label confusion in the dataset
- [ ] Expand the YOLO detection dataset for improved mAP
- [ ] Build a FastAPI backend wrapping the pipeline
- [ ] Build a React frontend for farmer-facing use
- [ ] Deploy backend (Hugging Face Spaces) and frontend (Vercel)
- [ ] Test on real, unseen farmer photos in field conditions

## License

MIT