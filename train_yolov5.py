import os
import yaml
import torch
import subprocess

# YOLOv5 저장소 클론 (이미 있다면 생략)
if not os.path.exists('yolov5'):
    subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'])

# # data.yaml 파일 생성
# data_yaml = {
#     'train': '/home/mihyun/CV_project/dataset_yolo/CV_Train1/Images',
#     'val': '/home/mihyun/CV_project/dataset_yolo/CV_Train1/Images',
#     'nc': 4,
#     'names': ['standing', 'lying', 'throwing', 'sitting']
# }

# with open('data.yaml', 'w') as f:
#     yaml.dump(data_yaml, f)

train_cmd = [
    'python', 'yolov5/train.py', 
    '--img', '640',
    '--batch', '16',
    '--epochs', '100',
    '--data', '/home/mihyun/CV_project/yolo.yaml',
    '--weights', 'yolov5s.pt',  # 작은 모델로 시작 (필요시 yolov5m.pt, yolov5l.pt, yolov5x.pt 사용 가능)
    '--device', '0',
    '--project', 'runs/train',
    '--name', 'yolov5'
]

# 학습 실행
subprocess.run(train_cmd)
