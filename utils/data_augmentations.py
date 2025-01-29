"""
This script is used to augment the data for the U-Net model. It reads the images and masks from the given path,
"""

import os
import cv2
import numpy as np
import imageio
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image

# Function to read a GIF file using Pillow and convert it to a format compatible with OpenCV
def gif_to_opencv(filepath):
    # Use Pillow to open the GIF file
    gif = Image.open(filepath)
    # Convert PIL image to numpy array
    frame = np.array(gif.convert('RGB'))
    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

def load_data(path, image_exts, mask_exts):
    # Load all image and mask paths
    all_imgs = sorted([img for ext in image_exts for img in glob(os.path.join(path, "images", "*" + ext))])
    all_masks = sorted([mask for ext in mask_exts for mask in glob(os.path.join(path, "masks", "*" + ext))])

    # Split data into training and testing sets
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(all_imgs, all_masks, test_size=0.5, random_state=42)

    return (train_imgs, train_masks), (test_imgs, test_masks)

def get_augmented_data(x, y):
    """
    Generate augmented data for a given image and mask.
    
    Args:
    - x (np.array): Input image.
    - y (np.array): Corresponding mask.
    
    Returns:
    - list: A list of augmented image-mask pairs.
    """
    x = cv2.imread(x)
    y = gif_to_opencv(y)
    rows, cols = x.shape[:2]
    augmentations = [
        (cv2.warpAffine(x, cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1), (cols, rows)), 
         cv2.warpAffine(y, cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1), (cols, rows)))
         for angle in [0,30,60,90,120,150,180]
    ]
    return augmentations

def augment_data(images, masks, save_path, size, augment=True):
    """
    Perform augmentation and save images and masks.
    
    Args:
    - images (list): List of paths to the images.
    - masks (list): List of paths to the masks.
    - save_path (str): Directory path to save the augmented images and masks.
    - size (tuple): Desired output size of the images and masks.
    - augment (bool): Whether to augment data or not. Default is True.
    """
    for x_path, y_path in tqdm(zip(images, masks), total=len(images)):
        name = os.path.basename(x_path).split(".")[0]
        x, y = cv2.imread(x_path, cv2.IMREAD_COLOR), imageio.mimread(y_path)[0]
        augmented = get_augmented_data(x, y) if augment else [(x, y)]
        for idx, (img, mask) in enumerate(augmented):
            cv2.imwrite(os.path.join(save_path, "images", f"{name}_{idx}.png"), cv2.resize(img, size))
            cv2.imwrite(os.path.join(save_path, "masks", f"{name}_{idx}.png"), cv2.resize(mask, size))

if __name__ == "__main__":

    images_path = sorted(glob("images/*"))
    mask_path =   sorted(glob("1st_manual/*"))
    

    save_images_path = "aug_images/"
    save_mask_path = "aug_labels/"

    for idx in range(len(images_path)):
        name = images_path[idx].split("/")[-1]
        name = os.path.splitext(name)[0]
        augmented = get_augmented_data(images_path[idx], mask_path[idx])
        for idx, (img, mask) in enumerate(augmented):
            cv2.imwrite(os.path.join(save_images_path, f"{idx}_{name}.jpg"), cv2.resize(img, (512,512)))
            cv2.imwrite(os.path.join(save_mask_path,f"{idx}_{name}.png"), cv2.resize(mask, (512,512)))
        

