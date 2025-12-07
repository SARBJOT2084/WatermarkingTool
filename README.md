# 🛡️ WatermarkLab | Robust Digital Forensics Tool

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)]()

> **University Project:** Digital Forensics & Image Processing  
> **Topic:** Robust Image Watermarking using Hybrid DWT-DCT & SIFT  
> **Reference Standards:** IEEE Transactions on Multimedia / Elsevier

---

## 📌 Project Overview
**WatermarkLab** is a forensic application capable of protecting digital images by embedding invisible copyright information. Unlike simple watermarking tools, this project implements a **Hybrid Frequency Domain (DWT-DCT)** approach combined with **Geometric Correction (SIFT)**.

This ensures the watermark survives severe attacks like **Rotation, Scaling, JPEG Compression, and Noise**, making it suitable for copyright protection and tamper detection.

### 🎯 Key Features
*   **Dual-Process Forensics:** Performs both *Action* (Embedding) and *Detection* (Extraction).
*   **Hybrid Embedding:** Uses Discrete Wavelet Transform (DWT) & Discrete Cosine Transform (DCT) to hide data in frequency bands.
*   **Geometric Defense:** Uses **SIFT (Scale-Invariant Feature Transform)** to automatically correct rotated or scaled images before extraction.
*   **Live Simulation:** Built-in attack lab to test robustness against JPEG (Q=50), Noise, and Rotation.
*   **Real-time Metrics:** Calculates **PSNR** (Image Quality) and **NC** (Robustness) instantly.

---

## 📂 Project Structure

```bash
WatermarkLab/
│
├── app.py                # Frontend: Streamlit Web Interface
├── backend.py            # Backend: Logic for DWT-DCT, SIFT, and Attacks
├── requirements.txt      # List of dependencies
├── README.md             # Project Documentation
└── assets/               # (Optional) Test images
