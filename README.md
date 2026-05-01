# 🦷 DenseNet121 Dental Caries Detection

A deep-learning pipeline that trains a **DenseNet121** binary classifier to detect dental caries (tooth decay) from dental X-ray images, with a **Streamlit** dashboard for interactive inference and Grad-CAM visualization.

---

## 📂 Repository Structure

```
.
├── densenet121-dental-caries.ipynb   # Training notebook (run on GPU to train)
├── app.py                            # Streamlit dashboard
├── requirements.txt                  # Python dependencies
├── dental_caries_dataset/            # Dataset (see "Dataset Setup" below)
│   └── final_dataset/
│       ├── train/
│       │   ├── caries/
│       │   └── healthy/
│       ├── val/
│       │   ├── caries/
│       │   └── healthy/
│       └── test/
│           ├── caries/
│           └── healthy/
└── training_results/                 # Generated after training (or extracted from ZIP)
    ├── densenet121_caries_best.pth   # Best model weights
    ├── metrics.json                  # Test-set metrics
    ├── training_history.json         # Per-epoch loss/accuracy
    ├── training_loss_curve.png       # Loss plot
    └── confusion_matrix.png          # Confusion-matrix heatmap
```

---

## ⚡ Quick Start

### 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2 — Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3 — Dataset Setup

The dataset must be placed (or symlinked) at:

```
dental_caries_dataset/final_dataset/
```

with the sub-folders `train/`, `val/`, `test/` each containing two class folders:
`caries/` and `healthy/`.

> **Note:** The dataset is **not** included in this repository due to size.  
> Download it from Kaggle (search *"teeth caries x-ray"*) and place it as shown above,  
> or follow the preprocessing pipeline described in the notebook.

### 4 — Get the trained weights (two options)

**Option A – Use pre-trained weights (recommended)**

Download `training_results.zip` from the [Releases](../../releases) page (or from the  
Kaggle notebook output), then extract it into the repo root:

```bash
# Windows (PowerShell)
Expand-Archive training_results.zip -DestinationPath .

# macOS / Linux
unzip training_results.zip
```

This creates the `training_results/` folder with all required files.

**Option B – Train from scratch**

Open `densenet121-dental-caries.ipynb` in Jupyter / Kaggle / Colab and run all cells  
(**GPU recommended**). After the final cell, `training_results.zip` is written to your  
working directory. Extract it as above.

### 5 — Launch the Streamlit app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

---

## 🏋️ Training Details

| Setting | Value |
|---------|-------|
| Backbone | DenseNet121 (ImageNet pre-trained) |
| Frozen layers | DenseBlocks 1–3 + all Transitions |
| Fine-tuned layers | DenseBlock 4 + Norm5 |
| Classifier head | Dropout(0.5) → Linear(1024 → 1) |
| Loss function | BCEWithLogitsLoss |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-3) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Imbalance handling | WeightedRandomSampler |
| Max epochs | 30 (early stopping patience=7) |
| Batch size | 16 |
| Input size | 224 × 224 RGB |

### Results on Test Set

| Metric | Value |
|--------|-------|
| Accuracy | 91.87 % |
| Caries Precision | 92.31 % |
| Caries Recall | 97.96 % |
| Healthy Precision | 89.47 % |
| Healthy Recall | 68.00 % |
| Weighted F1 | 91.44 % |

---

## 🔬 Streamlit App Pages

| Page | Description |
|------|-------------|
| 📊 Training Results | Loss/accuracy curves, confusion matrix, per-class metrics |
| 🔬 Predict & Grad-CAM | Upload an X-ray → prediction + Grad-CAM heatmap overlay |
| 🏗️ Model Architecture | Visual DenseNet121 architecture diagram |
| 📋 Report Analysis | Deep-dive: root-cause analysis of model performance |

---

## 📋 Requirements

See `requirements.txt`.  Key packages:

- `torch` / `torchvision` (CPU or CUDA)
- `streamlit`
- `pytorch-grad-cam`
- `scikit-learn`
- `matplotlib`, `seaborn`, `plotly`, `Pillow`

---

## 📄 License

MIT — see [LICENSE](LICENSE).
