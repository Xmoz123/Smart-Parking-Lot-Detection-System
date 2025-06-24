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

Here’s a cleaner, more polished version of your text:

---

The model was tested on the following types of images:
1. Real-world data
2. Images containing noise
3. Rainy conditions
4. Sunny conditions
5. Low-light or nighttime images

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

## License

This work is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.  
To view a copy of this license, visit [CC BY-NC-ND 4.0](http://creativecommons.org/licenses/by-nc-nd/4.0/).

**Please do not reuse, modify, or redistribute this code without my permission.**

---
## Contact

Feel free to connect with me on LinkedIn: (https://www.linkedin.com/in/pratheek-shanbhogue-54642231b/) for collaboration or discussion!
