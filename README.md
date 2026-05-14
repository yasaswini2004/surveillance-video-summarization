# A Visual-Language Agent for Summarizing Surveillance Footage Using Vision Transformers and Agentic LLM

## Overview

This project presents an end-to-end surveillance video summarization framework that converts CCTV footage into concise human-readable summaries using Spatio-Temporal Vision Transformers and Agentic Large Language Models.

The framework performs:
- Spatio-temporal feature extraction
- Temporal event detection
- Scene-level summarization
- End-to-end surveillance understanding

---

# System Architecture

## Overall Architecture

<img width="1570" height="414" alt="image" src="https://github.com/user-attachments/assets/5d4d303e-fa70-4cfb-8864-e720843a1fd5" />


---

## Spatio-Temporal Transformer Architecture

<img width="1214" height="346" alt="image" src="https://github.com/user-attachments/assets/2649dc32-4b06-45af-8f6a-977c0da3a91d" />


---

## Agentic LLM Architecture

<img width="537" height="377" alt="image" src="https://github.com/user-attachments/assets/8ae0a806-4685-444e-8736-2f5aee920133" />


---

# Methodology

## Stage 1 — Feature Extraction

Video clips are processed using VideoMAE to extract:
- spatial representations
- temporal motion dynamics
- contextual feature embeddings

The extracted embeddings are used for downstream temporal event detection.

---

### Stage 1 Output

<img width="457" height="385" alt="image" src="https://github.com/user-attachments/assets/86d39e1c-1b62-41eb-9e5f-8d14763ff2b2" />


---

### ROC Curves for Theft Classification

<img width="529" height="441" alt="image" src="https://github.com/user-attachments/assets/ba7a362e-0c41-41f8-9572-fb90e095726a" />


---

### VideoMAE Confusion Matrix

<img width="485" height="451" alt="image" src="https://github.com/user-attachments/assets/22306124-c36e-4840-8d62-30c27fd61c14" />


---

### Stage 1 Classification Performance Using VideoMAE

| Class | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Burglary | 0.28 | 0.28 | 0.28 | 0.526 |
| Robbery | 0.23 | 0.07 | 0.11 | 0.388 |
| Shoplifting | 0.11 | 0.14 | 0.13 | 0.480 |
| Stealing | 0.15 | 0.28 | 0.19 | 0.322 |
| Macro Avg | 0.19 | 0.19 | 0.18 | - |
| Weighted Avg | 0.21 | 0.18 | 0.18 | - |

---

# Stage 2 — Event Detection

Temporal event detection was performed using:
- TransformerClassifierV2
- BiLSTMClassifierV2

The models were trained on overlapping surveillance video segments generated from VideoMAE embeddings.

---

### Training Logs

<img width="1131" height="428" alt="image" src="https://github.com/user-attachments/assets/576305dc-c189-4ba8-ad0f-baace69cfd08" />


---

### Stage 2 Temporal Modeling Performance

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Transformer | 0.50 | 0.77 | 0.606 | 0.615 |
| BiLSTM | 0.49 | 0.78 | 0.607 | 0.609 |

Best Model:
- BiLSTMClassifierV2

---

# Stage 3 — Agentic LLM Summarization

Detected surveillance events are converted into coherent summaries using:
- FLAN-T5
- Generate → Reflect → Refine reasoning loop

---

### Sample Agentic LLM Summary

<img width="1410" height="373" alt="image" src="https://github.com/user-attachments/assets/65066c6b-9596-4130-b1ba-24cad09c8c2a" />

---

### Stage 3 Agentic LLM Summarization Performance

| Metric | Score |
|---|---|
| ROUGE-1 (F1) | 0.8490 |
| ROUGE-2 (F1) | 0.8360 |
| ROUGE-L (F1) | 0.8404 |
| BLEU-1 | 0.7438 |
| BLEU-2 | 0.7308 |

---

# Stage 4 — End-to-End Pipeline

The final pipeline integrates:
- frame extraction
- VideoMAE feature generation
- temporal event detection
- Agentic LLM summarization

to generate complete scene-level summaries for unseen surveillance videos.

---

### End-to-End Pipeline Output

<img width="1417" height="304" alt="image" src="https://github.com/user-attachments/assets/2b8f4b68-9482-4566-bd4d-e209d89b1da2" />


---

# GUI Testing Results

### GUI Output 1

<img width="825" height="582" alt="image" src="https://github.com/user-attachments/assets/f86adc89-7786-4eb4-b18b-ab1be23b7de4" />


---

### GUI Output 2

<img width="933" height="594" alt="image" src="https://github.com/user-attachments/assets/2297a883-4a33-429a-b1ab-1ecd94e615d6" />


---

# Technologies Used

| Category | Tools |
|---|---|
| Programming | Python 3.10 |
| Deep Learning | PyTorch, Torchvision |
| Vision Models | VideoMAE, ResNet50 |
| NLP Models | FLAN-T5 |
| Libraries | OpenCV, NumPy, PIL |
| Platform | Kaggle GPU |

---

# Dataset

The project was developed using theft-related categories from the UCF-Crime dataset:
- Burglary
- Robbery
- Shoplifting
- Stealing

---

# Experimental Setup

| Component | Specification |
|---|---|
| Processor | Intel i5 |
| RAM | 16GB |
| GPU | NVIDIA RTX 3060 |
| Training Platform | Kaggle Tesla T4/P100 |

---

# Project Structure

```text
surveillance-video-summarization/
│
├── notebooks/
│   └── final_notebook.ipynb
│
├── models/
├── requirements.txt
└── README.md
```

---

# Installation

Clone repository:

```bash
git clone https://github.com/YOUR_USERNAME/surveillance-video-summarization.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

```bash
python app.py
```

---

# Applications

- Smart surveillance systems
- Automated CCTV monitoring
- Public safety systems
- Smart city surveillance
- Incident summarization

---

# Future Scope

- Multi-camera tracking
- Real-time deployment
- Advanced anomaly localization
- Edge AI optimization
- Multi-modal surveillance understanding

---

# Acknowledgement

- IIT Tirupati Navavishkar I-Hub Foundation (IITTNiF)
- Geo-Intel Lab
- PNT Laboratory
- Siddhartha Academy of Higher Education

---

# Author

## Yasaswini Venuturumilli

AI & DS - B.Tech  
Siddhartha Academy of Higher Education  
Vijayawada, Andhra Pradesh, India

# Mentor

## Dr. Ashutosh Satapathy

Assistant Professor (Selection Grade) 
Dept. of CSE, Siddhartha Academy of Higher Education 
Vijayawada, Andhra Pradesh, India
