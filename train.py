import argparse

import numpy as np

import os
# os.environ["MKL_THREADING_LAYER"] = "GNU"

# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "42017"
# os.environ["WORLD_SIZE"] = "4"
# os.environ["RANK"] = "0"

import time

from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.xiilab.model import XiiYOLO


import warnings

# 특정 경고 메시지 무시
warnings.filterwarnings("ignore", message="Corrupt JPEG data")
warnings.filterwarnings("ignore")

def main(args):
    # Load a model
    # model = YOLO("yolo11x.pt")
    model = XiiYOLO("yolo11x.pt")

    # Train the model
    train_results = model.train(
        # data="coco8.yaml",  # path to dataset YAML
        # data="/workspace/data_path/DATASET/Competition_Dataset/CytologIA/yolo/data.yaml",
        data=args.data_path,
        epochs=int(args.epoch),  # number of training epochs
        imgsz=640,  # training image size
        device=args.gpu_num,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        # device=[0, 1],  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        save_period=5,
        batch=args.batch_size
    )

    # model = XiiYOLO("/DATA_17/pjw/workspace/ultralytics/runs/detect/train/weights/best.pt")

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Export the model to ONNX format
    # path = model.export(format="onnx")  # return path to exported model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=100, help='Epoch for Train')
    parser.add_argument('--gpu_num', default='2', type=str, nargs='+', help='0 1 ...')  # 공백으로 리스트 구현
    parser.add_argument('--batch_size', default=4, help='Batch Size for Train')
    # parser.add_argument('--data_path', default='coco8.yaml', help='Data for train')
    # parser.add_argument('--data_path', default='/DATA1/temp/data_tiny_balanced.yaml', help='Data for train')
    parser.add_argument('--data_path', default='/DATA/DATASETS/temp/data_tiny_balanced.yaml', help='Data for train')
    args = parser.parse_args()

    main(args)