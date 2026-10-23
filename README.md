# MarineVision — Underwater Intelligence & Sonar Analytics

**AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar (SSS) Imagery**

> **BTech Computer Engineering — 7th/8th Semester Capstone Project**  
> **Institute of Advanced Research (IAR University)**  
> **Project Group:** CodeQuacks (Group No. 55)  
> **Scholars:** Rudra Mistry (14311), Jaimin Thakkar (14159), Meet Patel (14486)  
> **Supervisor:** Ms. Pooja Chawla  

---

## 1. Project Overview & Architecture

**MarineVision** is an end-to-end artificial intelligence and analytics web application engineered specifically for Side-Scan Sonar (SSS) acoustic imagery. Underwater acoustic waterfall records pose significant challenges: severe speckle noise, acoustic shadows, beam attenuation, and complex seabed morphology. 

MarineVision solves these challenges with two complementary neural architectures built and trained **strictly from scratch without any pretrained weights or transfer learning**:
1. **Model 1: Marine Debris Detector (YOLO):** Multi-class bounding-box localization for underwater marine debris and navigation hazards (`submarine_pipeline`, `shipwreck`, `ghost_net`, `mine_cylinder`).
2. **Model 2: Seabed Anomaly Detector (CNN Autoencoder):** Deep convolutional reconstruction network trained exclusively on verified normal seabed textures to compute pixel-level error heatmaps and flag statistical anomalies (score vs. empirical validation threshold).

### Five-Page Application Structure

| Page # | Route | Purpose | Key Capabilities |
|---|---|---|---|
| **1** | `/intro` | System Introduction | Acoustic problem statement, project scope, architecture diagram, and pipeline overview. |
| **2** | `/` | Operational Dashboard | Real-time SQLite metrics, active model status cards, recent upload history, and pipeline benchmarks. |
| **3** | `/analyze`<br>`/analyze/<id>` | Combined Analysis Screen | Unified image upload, OpenCV acoustic preprocessing, dual-view comparison, real YOLO detections table, autoencoder anomaly heatmap, and ReportLab PDF download. |
| **4** | `/history` | Survey History & Audit | Filterable audit log of all scans, search by filename/ID, status indicators, and one-click PDF reports. |
| **5** | `/about` | Architectural Documentation | UML Level 1 & Level 2 architecture, 8-stage pipeline, academic credentials, and ML design rationale. |

---

## 2. Dataset Sources, Integrity & Licensing

The models are trained and benchmarked on genuine Side-Scan Sonar (SSS) acoustic datasets:
- **Source:** Hugging Face research repository `rehan9599/drishti-sss` (curated for SIH PS 26057: *"AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery"*).
- **License:** Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0).
- **Acoustic Modality:** True Side-Scan Sonar waterfall records exhibiting acoustic highlights and shadow zones (not optical underwater photos).
- **Zero Cross-Split Data Leakage:** Verified via cryptographic MD5 hashing. Any duplicate tiles appearing across splits were audited and removed.

### Object Detection Classes (YOLO)

Contiguous class mapping used in `configs/yolo_data.yaml`:
- `0: submarine_pipeline` — Underwater cables, conduits, and pipelines.
- `1: shipwreck` — Wreckage structures, submerged hulls, and debris fields.
- `2: ghost_net` — Abandoned, lost, or discarded fishing gear (synthetic netting backscatter).
- `3: mine_cylinder` — Cylindrical metallic hazards, munitions, and ordnance.

### Dataset Partitioning

```
data/
├── sonar_debris/            # YOLO Detection Dataset
│   ├── train/               # 369 images + normalized YOLO .txt annotations
│   ├── val/                 # 628 images + normalized YOLO .txt annotations
│   └── test/                # 698 held-out images + ground truth annotations
│
├── sonar_autoencoder/       # Unsupervised Anomaly Dataset
│   ├── train/               # 70 verified normal seabed tiles (empty target labels)
│   ├── val/                 # 68 verified normal seabed tiles (for threshold calibration)
│   └── test/
│       ├── normal/          # 69 verified normal seabed tiles
│       └── anomalous/       # 629 sonar images with actual debris/targets
│
└── dataset_manifest.json    # Complete cryptographic audit and provenance manifest
```

---

## 3. Training Pipelines (Trained Strictly from Scratch)

Both models are initialized with random weights and trained directly on the sonar datasets. **No pretrained checkpoints (`.pt`, `.pth`, `.onnx`), no transfer learning, and no external inference APIs are utilized.**

### 3.1 Model 1: YOLO Marine Debris Detector

- **Architecture:** Lightweight YOLOv8 Nano (`yolov8n.yaml`) configured with `nc: 4`.
- **Initialization:** Initialized from YAML architecture definition with `pretrained=False`.
- **Input Size:** 416x416 / 640x640 single-channel/RGB acoustic intensity.

**Train from scratch:**
```powershell
python train_yolo.py --arch yolov8n.yaml --data configs/yolo_data.yaml --epochs 15 --batch 8 --imgsz 416 --device cpu --output models/yolo_debris.pt
```

**Evaluate on held-out test set:**
```powershell
python evaluate_yolo.py --weights models/yolo_debris.pt --data configs/yolo_data.yaml --split test --imgsz 416 --device cpu
```
*Outputs: Precision, Recall, mAP@0.5, mAP@0.5:0.95, per-class breakdown saved to `models/yolo_evaluation.json`.*

### 3.2 Model 2: CNN Autoencoder Anomaly Detector

- **Architecture:** 4-stage convolutional encoder-decoder with latent bottleneck:
  - **Encoder:** `Conv2d(1->32) -> Conv2d(32->64) -> Conv2d(64->128) -> Conv2d(128->256) -> Latent(128)`
  - **Decoder:** `Linear -> ConvTranspose2d(256->128) -> ConvTranspose2d(128->64) -> ConvTranspose2d(64->32) -> Conv2d(32->1)`
- **Initialization:** Kaiming (He) normal initialization applied to all convolutional layers; biases set to zero.
- **Training Strategy:** Trained solely on verified normal seabed tiles (`MSELoss`, Adam optimizer, `lr=0.001`).
- **Anomaly Threshold Calibration:** Statistically derived from validation normal seabed reconstruction errors:
  $$\text{Threshold} = \mu_{\text{val}} + 2 \times \sigma_{\text{val}} = 0.1460 + 2(0.0507) = 0.2474$$

**Train from scratch:**
```powershell
python train_autoencoder.py --data-dir data/sonar_autoencoder --epochs 25 --batch 16 --lr 0.001 --device cpu --output models/autoencoder_anomaly.pth
```

**Evaluate on held-out test set (normal vs. anomalous):**
```powershell
python evaluate_autoencoder.py --weights models/autoencoder_anomaly.pth --data-dir data/sonar_autoencoder --split test --device cpu
```
*Evaluation Results on 698 Test Images: ROC-AUC: **0.7338**, Precision: **0.9812**, Specificity: **0.9565**.*

---

## 4. Flask Application Integration

The Flask inference pipeline loads the models on startup and executes both detectors sequentially:

1. **Preprocessing:** Edge-preserving bilateral filtering + Contrast-Limited Adaptive Histogram Equalization (CLAHE).
2. **Debris Detection (`predict_sonar`):** Evaluates preprocessed imagery through `detector.py`, generating bounding boxes, class labels, and confidence scores.
3. **Anomaly Detection (`detect_anomalies`):** Evaluates imagery through `anomaly_detector.py`, producing reconstruction error heatmaps, anomaly score, threshold comparison, and candidate region contours.
4. **Clean Seabed Differentiation:** When zero debris objects are detected, the system explicitly reports **"Seabed Area Clear — 0 Debris Detected"**, distinguishing a clean survey from a missing model.
5. **Database Persistence:** Stores runs, bounding boxes, anomaly scores, and processing times in SQLite (`analysis_runs`, `detections`, `anomalies`).
6. **Report Generation:** Generates a publication-grade ReportLab PDF report containing full survey audit trails and visual findings.

---

## 5. Quickstart & Installation

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Download & Prepare Sonar Datasets
```powershell
python scripts/fetch_training_data.py
python scripts/prepare_autoencoder_data.py
```

### Step 3: Run the Flask Web Application
```powershell
python app.py
```
Open your browser at: `http://127.0.0.1:5000`

---

## 6. Automated Testing

MarineVision includes two comprehensive test suites:

```powershell
# 1. Base Application & Security Test Suite (10 tests)
python -m unittest tests/test_suite.py -v

# 2. Comprehensive ML Architecture & Pipeline Suite (10 tests)
python -m unittest tests/test_ml_and_pipeline.py -v
```

Tests verify:
- Zero data leakage across splits.
- Random weight initialization from scratch (no pretrained weights).
- YOLO forward pass & bounding-box coordinate formats.
- Autoencoder reconstruction & error map dimensions.
- Graceful degradation when checkpoints are absent.
- SQLite ACID persistence & foreign key integrity.
- Vector-grade ReportLab PDF generation.

---

## 7. Checkpoints & Artifacts

All trained checkpoints and evaluation metrics are preserved in `models/`:
- `models/yolo_debris.pt` — Trained YOLOv8 Marine Debris Detector checkpoint.
- `models/yolo_evaluation.json` — YOLO test split performance metrics.
- `models/autoencoder_anomaly.pth` — Trained PyTorch Autoencoder checkpoint (includes calibrated threshold `0.2474`).
- `models/autoencoder_loss_curve.png` — Autoencoder training & validation MSE loss curves.
- `models/autoencoder_reconstructions.png` — Side-by-side normal seabed reconstruction pairs.
- `models/autoencoder_evaluation.json` — Autoencoder ROC-AUC, Precision, Recall, and confusion matrix.
- `models/autoencoder_evaluation_plots.png` — Reconstruction error distributions and ROC curve.

---

## 8. Academic Rationale & Ethical AI Standards

In accordance with academic research standards:
- **No Fabricated Predictions:** No hardcoded bounding boxes or simulated anomaly scores exist in the inference code. If a model is missing, the system explicitly states its status.
- **Clear Distinction of Findings:** High reconstruction error in the autoencoder indicates an unusual acoustic anomaly, not confirmed debris identity. Debris classifications are governed exclusively by YOLO detections.
- **Reproducibility:** All training scripts accept fixed random seeds (`--seed 42`) and generate complete audit logs.
