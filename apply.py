"""
this code is used to apply the trained model on the test images and save the results
"""


import glob
import os
import time

import cv2
import numpy as np
import torch
from models.nested_unet_model import NestedUNet
from models.sa_unet_model import SAUNet
from models.unet_model import UNet

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def preprocess(image_path,device):
    """ Reading image """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) ## (512, 512, 3)
    image_shape = image.shape[:2]
    image = cv2.resize(image, (512,512), interpolation=cv2.INTER_NEAREST_EXACT)
    x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
    x = x/255.0
    x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)
    return x, image_shape


if __name__ == "__main__":

    images_path = glob.glob("/workspace/images/*")




    model = UNet()
    model_path = glob.glob("/workspace/models/*.pth")[0]
    save_path = "/workspace/results/"

    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    count = 0
    for image_path in images_path:

        count += 1
        print(count)
        
        x, image_shape = preprocess(image_path,device)
        image_name = image_path.split("/")[-1].split(".JPG")[0] + ".png"
        # Calculate the final shape to be 50% of the original image shape
        final_height = image_shape[0] #// 2
        final_width = image_shape[1]  #// 2


        with torch.no_grad():
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            pred_y = pred_y[0].cpu().numpy()
        
        pred_y = np.squeeze(pred_y, axis=0)     
        pred_y = pred_y > 0.80
        pred_y = pred_y.astype(np.uint8) * 255

        # Resize to final shape in one step
        pred_y = cv2.resize(pred_y, (final_width, final_height), interpolation=cv2.INTER_NEAREST_EXACT)

        # Invert the image
        inverted_image = cv2.bitwise_not(pred_y)

        cv2.imwrite(save_path+image_name , inverted_image)
