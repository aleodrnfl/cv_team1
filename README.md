# 1-Stage. YOLOv5 Object Detection Project

이 프로젝트는 YOLOv5를 사용한 객체 감지 프로젝트입니다.

## 파일 구조
- `train_yolov5.py`: YOLOv5 모델 학습 스크립트
- `test_yolov5.py`: 학습된 모델 테스트 스크립트
- `yolo.yaml`: YOLOv5 설정 파일

## 사용 방법
1. 모델 학습:
```bash
python train_yolov5.py
```

2. 모델 테스트:
```bash
python test_yolov5.py
```

## 주의사항
- 모델 파일(`yolov5s.pt`)은 별도로 다운로드해야 합니다.
- 학습에 필요한 데이터셋은 별도로 준비해야 합니다. 



# 2-Stage model. YOLO for object detection and  MobileNet for action classification

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
