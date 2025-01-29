import os
import time
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, jaccard_score, precision_score, recall_score
)
from models.unet_model import UNet
from losses import create_dir, seeding


def calculate_metrics(y_true, y_pred):
    y_true = (y_true.cpu().numpy().reshape(-1) > 0.5).astype(np.uint8)
    y_pred = (y_pred.cpu().numpy().reshape(-1) > 0.5).astype(np.uint8)

    return [
        jaccard_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        accuracy_score(y_true, y_pred)
    ]


def mask_parse(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    return np.repeat(mask[..., np.newaxis], 3, axis=-1)

def load_data(path, target_size=None, color_mode="rgb"):
    if color_mode == "rgb":
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if target_size:
        img = cv2.resize(img, target_size)

    if len(img.shape) == 2:  # if grayscale
        img = img[..., np.newaxis]

    return img


def prepare_data(data, device):
    data = np.transpose(data, (2, 0, 1)) / 255.0
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    return data.to(device)


if __name__ == "__main__":
    
    seeding(42)

    TEST_DIR = "/test_data"
    create_dir(TEST_DIR+"/results")


    test_x = sorted(glob(TEST_DIR+"/test/images/*"))
    test_y = sorted(glob(TEST_DIR+"/test/masks/*"))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet().to(device)
    model.load_state_dict(torch.load("/test_data/checkpoint/*.pth", map_location=device))
    model.eval()

    metrics_scores = np.zeros(5)
    time_taken = []

    for x_path, y_path in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = os.path.basename(x_path).split(".")[0]

        x = load_data(x_path)
        y = load_data(y_path, color_mode="grayscale")

        x = prepare_data(x, device)
        y = prepare_data(y, device)

        with torch.no_grad():  # Add this
            start_time = time.time()
            pred_y = torch.sigmoid(model(x))
            time_taken.append(time.time() - start_time)

            metrics_scores += calculate_metrics(y, pred_y)

        pred_y_img = (pred_y[0].cpu().numpy().squeeze() > 0.5).astype(np.uint8)

        # Move tensors to CPU and convert to numpy arrays first
        x_np = x[0].permute(1, 2, 0).cpu().numpy()
        y_np = y[0].cpu().numpy().squeeze()

        combined_img = np.hstack([
            x_np, np.ones((512, 10, 3)) * 128, 
            mask_parse(y_np), np.ones((512, 10, 3)) * 128, 
            mask_parse(pred_y_img) * 255
        ])
        cv2.imwrite(TEST_DIR+f"/results/{name}.png", combined_img)

    metrics_scores /= len(test_x)
    print(f"Jaccard: {metrics_scores[0]:1.4f} - F1: {metrics_scores[1]:1.4f} - Recall: {metrics_scores[2]:1.4f} - Precision: {metrics_scores[3]:1.4f} - Acc: {metrics_scores[4]:1.4f}")
    print("FPS: ", 1/np.mean(time_taken))
