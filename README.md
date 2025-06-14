# YOLOv5 Object Detection Project

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