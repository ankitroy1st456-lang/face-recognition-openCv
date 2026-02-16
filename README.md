# Face Recognition with Python & OpenCV

## Overview
A simple face recognition system using Python and OpenCV.  
Pipeline: Capture → Train → Recognize.

## Requirements
opencv-contrib-python,
numpy,
pillow

# Usage
1. `create_user(1, "roy")` → capture dataset  
2. `train()` → train model  
3. `recognize({1: "roy"})` → run recognition