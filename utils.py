
import os
import random
import numpy as np
import cv2
import torch
from glob import glob


""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


from PIL import Image, ImageSequence

def gif_to_opencv_image(gif_path):
    """
    Read the first frame of a GIF and convert to OpenCV format.
    """
    with Image.open(gif_path) as img:
        # Convert the first frame from PIL to OpenCV format
        opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
    return opencv_img


def resize_image(path, size):
    images = sorted(glob(path))
    print("Resizing images...")
    
    for image in images:
        # img = cv2.imread(image)
        img = gif_to_opencv_image(image)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        
        # check if the extension is .gif or .png and remove both
        if image.split("/")[-1].split(".")[-1] == "gif":
            name = "/train_masks_with_drive/"+ image.split("/")[-1].replace(".gif",".png")
        else:
            name = "/train_masks_with_drive/"+ image.split("/")[-1]

        cv2.imwrite(name, img)

# if main
if __name__ == "__main__":
    path = "/train_masks_with_drive/*"
    size = 512
    resize_image(path, size)        