# Smart Parking Lot Detection System

This project is a **two-stage machine learning model** for parking lot identification and detection.  
It identifies parking slots in images and classifies them as **empty** or **occupied**.

---

## Purpose

This repository is intended to demonstrate my machine learning and computer vision skills to recruiters, peers, and collaborators.  
⚠ **Please note:** This code is for viewing only. Reuse, modification, or redistribution is not permitted without my permission.

---

## Technologies Used
- YOLOv8 (Stage 1: Parking slot detection)
- Convolutional Neural Network (Stage 2: Slot classification)
- Python  
- OpenCV  
- PyTorch

---

## Performance
| Stage | Model | Accuracy | mAP50-95 |
|--------|--------|----------|----------|
| 1 | YOLOv8 | ~99% (based on precision-recall, confidence matrix) | 0.941 |
| 2 | CNN | 98.83% | N/A |

---

## Project Structure
- `stage 1 scripts/` → YOLOv8 detection code  
- `stage 2 scripts/` → CNN classification code  
- `test_images/` → Sample input images  
- `test_results/` → Model outputs  
- `csv_output/` → CSV files with parking slot status  
- `performance analysis of both stages/` → Metrics, curves, charts  
- `Read me` → Summary info  

---

## Contact

Feel free to connect with me on LinkedIn: (https://www.linkedin.com/in/pratheek-shanbhogue-54642231b/) for collaboration or discussion!
