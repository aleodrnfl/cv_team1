import torch
import os
import json
from glob import glob
from tqdm import tqdm

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/mihyun/CV_project/runs/train/yolov5/weights/last.pt')

# 테스트 데이터 경로
test_dir = '/home/mihyun/CV_project/CV_Test/Images'
output_dir = '/home/mihyun/CV_project/CV_Test/Labels_yolov5'
os.makedirs(output_dir, exist_ok=True)

# 클래스 이름 정의
class_names = ['standing', 'lying', 'throwing', 'sitting']

# 테스트 이미지 파일 리스트
test_images = sorted(glob(os.path.join(test_dir, '*.png')))

print(f"전체 테스트 이미지: {len(test_images)}개")

# 각 이미지에 대해 예측 수행
for img_path in tqdm(test_images, desc="테스트 진행 중"):
    # 이미지 파일명 (확장자 제외)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # 예측 수행
    results = model(img_path)
    
    # 결과를 JSON 형식으로 변환
    json_data = {
        "version": "5.8.1",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(img_path),
        "imageData": None,
        "imageHeight": results.ims[0].shape[0],
        "imageWidth": results.ims[0].shape[1]
    }
    
    # 검출된 객체들에 대해 처리
    for *xyxy, conf, cls in results.xyxy[0]:
        # 바운딩 박스 좌표 (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(float, xyxy)
        
        # 클래스 ID와 신뢰도
        class_id = int(cls)
        confidence = float(conf)
        
        # JSON 형식으로 변환
        shape = {
            "label": class_names[class_id],
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None
        }
        json_data["shapes"].append(shape)
    
    # JSON 파일 저장
    output_path = os.path.join(output_dir, f"{img_name}.json")
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)

print(f"테스트 완료! 결과는 {output_dir} 폴더에 저장되었습니다.") 