import argparse
import os
import warnings
from ultralytics import YOLO
from ultralytics.xiilab.model import XiiYOLO
import torch.distributed as dist
import torch
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "42017"
os.environ["WORLD_SIZE"] = "3"
os.environ["RANK"] = "0"

# def setup_distributed():
#     """Set up distributed training."""
#     dist.init_process_group(backend="nccl", init_method="env://")
#     local_rank = int(os.environ["LOCAL_RANK"])
#     torch.cuda.set_device(local_rank)
#     return local_rank

# def cleanup_distributed():
#     """Clean up distributed training."""
#     dist.destroy_process_group()

def main(args):
    # Set up distributed training
    # local_rank = setup_distributed()

    # Initialize model
    model = XiiYOLO("yolo11x.pt")

    # Train the model
    train_results = model.train(
        data=args.data_path,  # path to dataset YAML
        epochs=int(args.epoch),  # number of training epochs
        imgsz=640,  # training image size
        device=[0,1,2],  # specific GPU assigned to the process
        save_period=5,
        batch=args.batch_size
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Clean up distributed training
    # cleanup_distributed()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, help='Epoch for Train')
    parser.add_argument('--batch_size', default=12, help='Batch Size for Train')
    parser.add_argument('--data_path', default='/DATA1/temp/data_all.yaml', help='Data for train')
    args = parser.parse_args()

    main(args)
