# 1-Stage. YOLOv5 Object Detection Project

This project focuses on object detection using YOLOv5.

## File Structure
* `train_yolov5.py`: Script for training the YOLOv5 model.
* `test_yolov5.py`: Script for testing the trained model.
* `yolo.yaml`: Configuration file for YOLOv5.

## Setup
1. Install the required packages:
   ```
   pip install torch tqdm pyyaml
   ```

2. Train YOLOv8 Detection Model:
```bash
python train_yolov5.py
```

3. Test YOLOv8 and change format to json:
```bash
python test_yolov5.py
```

## Important Notes

* The pre-trained model file (`yolov5s.pt`) must be downloaded separately.
* The dataset required for training must be prepared separately. <br>
<br>


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
- yolov8x.pt (xlarge) - slowest but most accurate<br>
<br>

# K-Fold Data Filtering for Robust Model Training

To enhance the robustness and accuracy of our action classification model, we employ a K-Fold Data Filtering strategy inspired by K-Fold cross-validation. This method helps us identify and exclude noisy or unrepresentative data, leading to improved model optimization.

## Motivation

* **Inspired by K-Fold cross-validation:** We assess the classification accuracy of each image to understand its contribution to the overall model performance.
* **Addressing data outliers:** We assume that images with low classification accuracy are likely outliers or apart from the true data distribution. By removing these, we prevent them from negatively impacting model optimization.

## Method

The K-Fold Data Filtering process involves the following steps:

1.  **Divide the data into K subsets:** The entire dataset is partitioned into K equally sized folds.
2.  **Train the model K times:** The model is trained K separate times. In each iteration, K-1 folds are used for training, and the remaining 1 fold is reserved for testing. This allows us to evaluate the model's performance on different subsets of the data.
3.  **Discard images with lower accuracy:** Based on the evaluations from the K training runs, images that consistently yield lower classification accuracy than a pre-defined threshold are identified and discarded. This ensures that only high-quality data contributes to the final model.
4.  **Train the final action classification model (MobileNet) with filtered dataset:** After the filtering process, the MobileNet model is trained one last time using the refined, higher-quality dataset. This final training benefits from the removal of potentially misleading or noisy samples.

## Related Files

The following files are integral to the implementation of the K-Fold Data Filtering process:

* `train_mobilenetv3_kfold.py`: Script responsible for training the MobileNetV3 model using the K-Fold cross-validation strategy, including the accuracy assessment for each image.
* `gen_action_crops_kfold.py`: Generates image crops relevant to actions, likely preparing them for K-Fold processing and further analysis.
* `convert_label_to_yolo_kfold.py`: Converts annotation labels (possibly from LabelMe or similar) into YOLO format, tailored for the K-Fold pipeline.
* `mobilenet_kfold_image_scores.csv`: Stores the classification scores or accuracies for each image across the K-Fold runs, used for identifying low-accuracy images.
* `per_class_accuracy.csv`: Provides a breakdown of accuracy per class, useful for understanding classification performance at a granular level.
* `per_image_accuracy.csv`: Contains the individual classification accuracy for each image, which is crucial for the filtering step.
