# Cricket Ball Detection & Tracking System 🏏

A computer vision pipeline built for the **EdgeFleet.AI Assessment**. This system detects and tracks cricket balls in video footage using **YOLOv8** and generates trajectory analytics.

## 📌 Project Overview

- **Goal:** Detect ball centroid and track trajectory in single-camera footage.

- **Model:** Custom trained YOLOv8 Nano.

- **Input:** Raw MP4/MOV cricket videos.

- **Output:** Processed video with overlay + CSV annotation file.

## 📂 Repository Structure

EdgeFleet_Assessment/
├── code/
│   ├── train.py          # Training script (Transfer Learning with YOLOv8)
│   ├── inference.py      # Main pipeline (Detection + Tracking + Smoothing)
│   └── create_subset.py  # Utility script for dataset management
│
├── annotations/          # Generated CSV files (tracking & smoothing outputs)
├── results/              # Output processed videos
├── models/               # Trained YOLOv8 weights (best.pt)
│
├── Report.pdf            # Detailed technical report
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation


## 🚀 Setup & Usage

### 1. Installation

Clone the repository and install dependencies:

```bash

git clone https://github.com/Aditya-164/Edgefleet_Assignment.git

cd Edgefleet_Assignment

pip install -r requirements.txt
2. Inference (Running on Test Videos)
To generate annotations and tracking videos for the test set:

Place input videos in a folder named test_videos/ (outside the repo to save space) or update path in inference.py.
Run the script:

cd code

python inference.py

Results will be saved to results/ and annotations/.
3. Training (Reproducibility)
To replicate the training process:

Download the "Cricket Ball" dataset from Roboflow (YOLOv8 format).
Update dataset/data.yaml paths.
Run:

cd code

python train.py
🧠 Methodology
Data Strategy: Trained on an external open-source dataset (1,500 images) to prevent data leakage from test videos.
Filtering: Implemented Spatial Consistency Checks to reject false positives (e.g., white shoes) based on unrealistic movement speed.
Smoothing: Uses linear interpolation to fill missing detections during motion blur.
📝 Limitations
Trained on CPU with a subset of data. Full GPU training on the complete dataset would further improve accuracy.
Simple physics-based tracking used; Kalman filters recommended for production V2.

---

