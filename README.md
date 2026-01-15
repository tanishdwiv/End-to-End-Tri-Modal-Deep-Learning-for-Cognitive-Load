
# 🧠 End-to-End Tri-Modal Deep Learning for Cognitive Load Estimation

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN%20%7C%20LSTM%20%7C%20BiLSTM-orange)
![Status](https://img.shields.io/badge/Status-Research%20Project-success)
![License](https://img.shields.io/badge/License-Academic-lightgrey)
![Dataset](https://img.shields.io/badge/Dataset-CLARE-purple)

> 🎓 Research project implementing an **attention-based tri-modal deep learning framework** for real-time cognitive load estimation using **EEG, physiological signals, and gaze data**.

---

## 📌 Overview

This project implements the research paper:

> **End-to-End Tri-Modal Deep Learning for Cognitive Load: CNN–BiLSTM EEG, CNN–LSTM Physiology, and Dense Gaze Fusion on CLARE**

It proposes a **multimodal AI system** that combines:

* 🧠 EEG signals
* ❤️ Physiological signals (ECG, EDA)
* 👁️ Eye-gaze features

Using:

* CNN + BiLSTM (EEG encoder)
* CNN + LSTM (Physiology encoder)
* Dense neural layers (Gaze encoder)
* Attention-based fusion for improved performance

---

## 🚀 Key Features

✔ End-to-end multimodal deep learning pipeline
✔ Attention-based cross-modal fusion
✔ Leave-One-Subject-Out (LOSO) evaluation
✔ Detailed ablation studies (unimodal, bimodal, trimodal)
✔ Real-world applicability (HCI, adaptive systems, monitoring)

---

## 📊 Results

| Model                               | Accuracy   |
| ----------------------------------- | ---------- |
| EEG only                            | 65.2%      |
| EEG + Physiology                    | 71.4%      |
| Trimodal (no attention)             | 74.8%      |
| **Trimodal + Attention (Proposed)** | **78.33%** |

**Final Metrics (Best Model):**

* Accuracy: **78.33%**
* Precision: **72.36%**
* Recall: **78.33%**
* F1-score: **73.67%**

---

## 📂 Dataset

This project uses the **CLARE Dataset (Cognitive Load Assessment in REaltime)**:

🔗 [https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/H0AELT](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/H0AELT)

> ⚠️ Dataset is **not included** in this repository due to size and licensing.
> Please download it manually and place it in a `data/` directory.

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras / PyTorch (whichever you used)
* NumPy, Pandas
* Scikit-learn
* Matplotlib / Seaborn
* Deep Learning (CNN, LSTM, BiLSTM, Attention)

---

## 📦 Project Structure (example)

```
├── train.py
├── visualisation.py
├── models/
├── utils/
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

```bash
git clone https://github.com/tanishdwiv/End-to-End-Tri-Modal-Deep-Learning-for-Cognitive-Load.git
cd End-to-End-Tri-Modal-Deep-Learning-for-Cognitive-Load
pip install -r requirements.txt
python train.py
```

---

## 🎯 Applications

* Adaptive learning systems
* Driver monitoring systems
* Human-computer interaction (HCI)
* Cognitive state monitoring
* Human–AI teaming

---

## 👨‍💻 Authors

* **Tanish Dwivedi**
* Jyothika Rajesh
* Dr. Gaurav Agarwal
  School of Computer Science and Engineering, Galgotias University

---

## 📜 Citation

If you use this project, please cite:

> Rajesh, J., Dwivedi, T., Agarwal, G. (2025). End-to-End Tri-Modal Deep Learning for Cognitive Load Estimation using CLARE Dataset.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!



Just tell me which style you want.
