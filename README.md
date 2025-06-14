# 2-Stage. YOLO for object detection and  MobileNet for action classification

# YOLO Action Detection

This project implements action detection (standing, sitting, lying, throwing) using YOLOv8 and the Ultralytics library.

Adjust script arguments (e.g., `--epochs`, `--model`, `--device`, other path... ) as needed based on your setup and performance requirements!

## Setup

1. Install the required packages:
   ```
   pip install ultralytics opencv-python matplotlib torch timm wandb
   ```

2. Convert LabelMe JSON to YOLO Format:
   ```
   python convert_label_to_yolo.py
   ```

3. Generate Action Classification Crops:
   ```
   python gen_action_crops.py
   ```

4. Train YOLOv8 Detection Model:
   ```
   python train_yolo.py
   ```

5. Train MobileNetV3 Action Classifier:
   ```
   python train_mobilenetv3.py
   ```

6. Run Inference and Generate Submission JSONs:
   ```
   python gen_submission_json.py
   ```

7. Create zip file for Submission
   ```
   zip -r submission_json.zip submission_json
   ```

## Automated Pipeline

Run the entire training and evaluation pipeline automatically:
```
./run_all.sh
```

## Scripts Description


- **convert_label_to_yolo.py**  
  Converts LabelMe-style JSON annotations to YOLO format.

- **gen_action_crops.py**  
  Generates cropped human action images for classification.

- **train_yolo.py**  
  Trains a YOLO model on the person-detection dataset.

- **train_mobilenetv3.py**  
  Trains a MobileNetV3 model for action classification.

- **gen_submission_json.py**  
  Runs YOLO detection + classification and generates submission JSONs.

- **run_all.sh**  
  Automates the entire pipeline from preprocessing to final submission.

## Dataset Structure

The dataset contains images with bounding boxes for four action classes:
- standing
- sitting
- lying 
- throwing

## Model Options

You can choose from various YOLOv8 model sizes:
- yolov8n.pt (nano) - fastest but less accurate
- yolov8s.pt (small)
- yolov8m.pt (medium)
- yolov8l.pt (large)
- yolov8x.pt (xlarge) - slowest but most accurate

1-Stage YOLOv5 Object Detection Project
This repository contains a simple object detection pipeline using YOLOv5.

📁 File Structure
train_yolov5.py – Script to train the YOLOv5 model

test_yolov5.py – Script to test the trained model

yolo.yaml – YOLOv5 configuration file (e.g., dataset and class info)

🛠️ Usage
1. Train the model
bash
복사
편집
python train_yolov5.py
2. Test the model
bash
복사
편집
python test_yolov5.py
⚠️ Notes
The YOLOv5 weights file (e.g., yolov5s.pt) must be downloaded separately from the Ultralytics YOLOv5 repository.

A properly formatted dataset (images and labels) must be prepared in advance. You can follow the YOLOv5 data format guide for more information.