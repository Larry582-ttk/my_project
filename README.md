# 🏗️ Machine Learning / Deep Learning Portfolio

<p align="center">
  <b>Larry</b> ·
  City University of Hong Kong ·
  Surveying (BSc) ·
  GitHub
</p>

---

Hi, I'm Larry. I'm an undergraduate student in Surveying at City University of Hong Kong, currently self-studying machine learning and deep learning. My projects focus on data-driven applications in construction safety, urban monitoring, infrastructure management, and related scenarios, which are areas I plan to further explore during my master's studies (in Digital Construction / Smart City / Geospatial Data Science).

> **Disclaimer:** This project is for personal learning and portfolio demonstration purposes only. The code implementation references public tutorials, official documentation, community blogs, and AI-assisted tools. All datasets used are obtained through open-source channels. The project code is for educational and non-commercial use only.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Larry582-ttk/my_project
cd my_project

# Install dependencies
pip install -r requirements.txt
```

| Project | Run Command |
| :--- | :--- |
| Construction Site Accident Risk Classification | `cd hk_construction_risk && python logistic_model.py` |
| Motorcycle Helmet Detection | `cd helmet_detection && python main.py` |
| LSTM Traffic Flow Prediction | `cd lstm && python main.py` |

---

## Project List

| # | Project | Domain | Core Techniques |
| :-: | :--- | :--- | :--- |
| 1 | Construction Site Accident Risk Classification | NLP + Machine Learning | Sentence-BERT, Hierarchical Clustering, Logistic Regression, Decision Tree |
| 2 | Motorcycle Helmet Detection | Computer Vision | VGG16, ONNX Export |
| 3 | LSTM Traffic Flow Prediction | Time Series | LSTM, Sliding Window |

---

## Project 1: Construction Site Accident Risk Classification

📁 `hk_construction_risk/`

Automatically classify accident titles from Hong Kong Labour Department Work Safety Alerts into 9 risk categories. Employs a two-stage strategy: unsupervised clustering for rapid labeling, followed by supervised classifier training for new data prediction.

### Data

| Item | Description |
| :--- | :--- |
| **Source** | Hong Kong Labour Department website — Work Safety Alerts |
| **Time Range** | 2012–2024 |
| **Size** | 398 accident titles |
| **Collection Method** | Web scraping + manual organization |

### 9-Class Classification System

| Code | Category |
| :--: | :--- |
| 1 | Fall from Height |
| 2 | Struck by Falling Object |
| 3 | Struck by Vehicle / Moving Object |
| 4 | Trapped / Crushed / Caught / Collapse |
| 5 | Electrocution |
| 6 | Exposure / Asphyxiation / Poisoning |
| 7 | Fire / Explosion / Burn |
| 8 | Drowning |
| 0 | Others |

### Pipeline

```text
Web Scraping → Sentence-BERT Embedding → Hierarchical Clustering
→ Manual Labeling (189 samples) → Train Classifier (Logistic Regression / Decision Tree) → Predict New Test Data
```

### File Structure

```text
hk_construction_risk/
├── logistic_model.py              # Logistic Regression model
├── decisiontree_model.py          # Decision Tree model
├── web_scraper.py                 # Web scraper
├── pca_exploration.py             # PCA dimensionality reduction exploration (optional)
├── data/
│   ├── raw/                       # Raw data (including training / test sets)
│   ├── intermediate/              # To-be-labeled / labeled data
│   └── output/                    # Classification results
└── models/                        # Saved models
```

### Tech Stack

`requests`, `BeautifulSoup`, `sentence-transformers`, `scikit-learn`, `joblib`

### Results (398 labeled samples)

| Category | Count | Percentage |
| :--- | :-: | :-: |
| Fall from Height | 170 | 42.7% |
| Trapped / Crushed / Caught / Collapse | 91 | 22.9% |
| Struck by Vehicle / Moving Object | 44 | 11.1% |
| Struck by Falling Object | 33 | 8.3% |
| Electrocution | 21 | 5.3% |
| Exposure / Asphyxiation / Poisoning | 15 | 3.8% |
| Others | 11 | 2.8% |
| Drowning | 8 | 2.0% |
| Fire / Explosion / Burn | 5 | 1.3% |

> **Fall from Height (42.7%) and Trapped/Collapse (22.9%) together account for over 65% of all incidents**, making them the most fatal accident types at Hong Kong construction sites.

---

## Project 2: Motorcycle Helmet Detection

📁 `helmet_detection/`

A simple object detection task based on VGG16, simultaneously predicting the location and class of a single object in an image. Only the first annotated object per image is used for training, outputting bounding box coordinates (x, y, w, h) and 4 class probabilities.

### Data

| Item | Description |
| :--- | :--- |
| **Task Type** | Simple object detection (location regression + 4-class classification) |
| **4 Classes** | `no_helmet`, `motor`, `number`, `with_helmet` |
| **Data Format** | YOLO format (txt) (VOC format loading code is also implemented; current training script uses YOLO format) |
| **Dataset Size** | 144 training images, 18 validation images |

### Method

Uses VGG16 as the backbone network, replacing the final layer with a fully connected layer of 8 outputs (first 4 for bounding box coordinates, last 4 for class probabilities), trained with SGD optimizer. The loss function consists of location loss (MSE) and classification loss (CrossEntropy). After training, the model is exported to ONNX format.

### Pipeline

```text
YOLO Data Loading → Image Preprocessing → VGG16 Feature Extraction
→ 8-Output FC Layer (4 coordinates + 4 classes) → SGD Training → ONNX Export
```

### File Structure

```text
helmet_detection/
├── main.py                        # Main training script
├── model.py                       # VGG16 + 8-output FC layer (4 coords + 4 classes)
├── loss_fn.py                     # Loss function (MSE + CrossEntropy)
├── utils/
│   ├── load_yolo_dataset.py       # YOLO format data loader
│   └── load_voc_dataset.py        # VOC format data loader
├── dataset/
│   ├── HelmetDataset-VOC/         # VOC format dataset
│   ├── HelmetDataset-YOLO-Train/  # YOLO training set
│   └── HelmetDataset-YOLO-Val/    # YOLO validation set
└── outputs/                       # Model output directory
```

### Tech Stack

`PyTorch`, `torchvision`, `ONNX`, `NumPy`

**Acknowledgments:** This is a learning/reproduction project. The code implementation references [xiaotudui](https://github.com/xiaotudui/tudui-object-detection-model)'s YOLO tutorial (MIT License). Special thanks to the original author.

---

## Project 3: LSTM Traffic Flow Prediction

📁 `lstm/`

Uses LSTM to predict node traffic flow based on California highway traffic flow data.

### Data

| Item | Description |
| :--- | :--- |
| **Dataset** | PeMSD4 (Caltrans Performance Measurement System) |
| **Number of Nodes** | 307 sensor nodes |
| **Original Features** | flow, occupy, speed — only flow is used for univariate prediction |
| **Time Granularity** | 5 minutes / sample point |
| **Selected Node** | node_id = 1 |
| **Sample Construction** | Sliding window (history = 12 → future = 1), using StandardScaler to normalize features and labels separately |

### Method

A two-layer unidirectional LSTM (hidden dimension 64), split chronologically into training set (first 80%) and test set (last 20%). Trained for 100 epochs, batch_size = 128, learning rate = 0.001, using Adam optimizer and MSELoss as the loss function.

### Results

| Metric | Value |
| :--- | :-: |
| R² | 0.9167 |
| MAE | 24.12 |

> The model explains **91.67%** of the data variance.

### Pipeline

```text
Raw Data (use only flow for prediction) → StandardScaler Normalization → Sliding Window Sample Construction (12→1)
→ Chronological Split (8:2) → LSTM Training → R² / MAE Evaluation → Result Visualization
```

### File Structure

```text
lstm/
├── main.py                        # Main training script
├── model.py                       # LSTM model
├── loss_fn.py                     # Loss function
├── utils/data_utils.py            # Data loading & preprocessing (normalization, sliding window sample construction)
├── dataset/pems04.npz             # Traffic flow dataset
└── outputs/                       # Result figures (prediction comparison + scatter plot)
```

### Tech Stack

`PyTorch`, `NumPy`, `pandas`, `scikit-learn`, `Matplotlib`

---

## Tech Stack Overview

| Area | Tools |
| :--- | :--- |
| Machine Learning / Deep Learning | PyTorch, scikit-learn, sentence-transformers |
| Data Processing & Visualization | pandas, NumPy, Matplotlib |
| Other Tools | Git, ONNX, joblib, BeautifulSoup |

---

<p align="center">
  © 2026 Larry · Portfolio
</p>
