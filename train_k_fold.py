import cProfile
import os
import pstats
import random
from glob import glob

import cv2
import h5py
import numpy as np
import torch
# import the losses 
from losses.nested_unet_loss import BCEDiceLoss, DiceBCELoss
# import the models 
from models.nested_unet_model import NestedUNet
from models.sa_unet_model import SAUNet
from models.unet_model import UNet
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def crop_512(image):
    # Get original dimensions
    h, w = image.shape[:2]
    
    # Calculate the new dimensions to have the short side as 512 while keeping the aspect ratio
    if h < w:
        new_h = 512
        new_w = int(w * (512 / h))
    else:
        new_w = 512
        new_h = int(h * (512 / w))
    
    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Calculate the center crop coordinates
    center_x, center_y = new_w // 2, new_h // 2
    half_crop_size = 512 // 2
    
    start_x = center_x - half_crop_size
    start_y = center_y - half_crop_size
    
    # Crop the image to 512x512 from the center
    cropped_image = resized_image[start_y:start_y + 512, start_x:start_x + 512]
  
    return cropped_image

def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_images(image_paths, type="image"):
    images = []
    if type == "image":
        # read as color
        for image_path in image_paths:
                """ Reading image """
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                images.append(image)
    else:# mask
        # read as grayscale
        for image_path in image_paths:
            """ Reading image """
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
    return images

def preprocess(all_images,type="image"):
    images = []
    if type == "image":
        for image in all_images:
            """ Reading image """
            image = crop_512(image.copy()) # crop 512*512 if you afraid that resizing will cause distortion
            image = image/255.0 ## (512, 512, 3)
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = torch.from_numpy(image)
            images.append(image)  
        return images    

    elif type == "mask":
        for mask in all_images:
            """ Reading mask """
            mask = crop_512(mask.copy()) # crop 512*512 if you afraid that resizing will cause distortion
            mask = mask/255.0
            # apply binary thresholding 
            mask[mask > 0] = 1
            mask = np.expand_dims(mask, axis=0)
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask)
            images.append(mask)
            
        return images
    else:
        raise ValueError("type must be image or mask")    

class UNetTrainer:
    def __init__(self, config, model):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)       
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['LEARNING_RATE'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=self.config['PATIENCE'], verbose=True)
        self.loss_fn = BCEDiceLoss()
        self.scaler = GradScaler()

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0 
        correct_preds = 0.0 
        total_samples = 0
        iou_sum = 0.0
        dice_sum = 0.0

        for x, y in loader:
            x, y = x.to(self.device, dtype=torch.float32), y.to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad()
            with autocast():
                y_pred = self.model(x)
                loss, dice, iou  = self.loss_fn(y_pred, y)
            self.scaler.scale(loss).backward()  
            self.scaler.step(self.optimizer)
            self.scaler.update()
            preds = torch.sigmoid(y_pred)
            correct_preds += (preds > 0.5).float().eq(y).sum().item() 
            total_samples += y.numel()
            total_loss += loss.item()
            dice_sum += dice.item()
            iou_sum += iou.item()

        acc = correct_preds / total_samples
        return total_loss / len(loader), acc, dice_sum / len(loader), iou_sum / len(loader)
    
    def evaluate_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        correct_preds = 0
        total_samples = 0
        dice_sum = 0.0
        iou_sum = 0.0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device, dtype=torch.float32), y.to(self.device, dtype=torch.float32)

                with autocast():
                    y_pred = self.model(x)
                    loss, dice, iou = self.loss_fn(y_pred, y)
                preds = torch.sigmoid(y_pred)
                correct_preds += (preds > 0.5).float().eq(y).sum().item()
                total_samples += y.numel()
                total_loss += loss.item()
                dice_sum += dice.item()
                iou_sum += iou.item()

        acc = correct_preds / total_samples
        return total_loss / len(loader), acc, dice_sum / len(loader), iou_sum / len(loader)

    def train(self, train_loader, valid_loader, fold):
        best_train_acc = 0
        best_valid_acc = 0
        best_valid_loss = float("inf")
        best_dice = 0.0
        best_iou = 0.0

        for epoch in tqdm(range(self.config['EPOCHS']), desc=f"Fold {fold+1} Progress"):
            train_loss, train_acc,train_dice, train_iou = self.train_epoch(train_loader)
            valid_loss, valid_acc , valid_dice, valid_iou  = self.evaluate_epoch(valid_loader)
            self.scheduler.step(valid_loss)

            if valid_loss < best_valid_loss:    
                best_valid_loss = valid_loss
                best_valid_acc = valid_acc
                best_train_acc = train_acc
                best_dice = valid_dice
                best_iou = valid_iou
                filename = f"fold_{fold+1}_epoch_{epoch+1}_valid_loss_{valid_loss:.3f}_valid_acc_{valid_acc:.3f}_train_acc_{train_acc:.3f}_best_dice_{best_dice:.3f}_best_iou_{best_iou:.3f}.pth"
                torch.save(self.model.state_dict(), self.config['CHECKPOINT_PATH'] + filename)

        return best_train_acc, best_valid_acc,best_valid_loss,best_dice, best_iou


def load_data(path):
    with h5py.File(path, 'r') as hf:
        all_images_tensor = torch.from_numpy(hf['images'][:])
        all_masks_tensor = torch.from_numpy(hf['masks'][:])

    return all_images_tensor, all_masks_tensor

def save_data(path,all_images, all_masks):

    with h5py.File(path, 'w') as hf:
        all_images_np = np.stack([img.numpy() for img in all_images])
        all_masks_np = np.stack([mask.numpy() for mask in all_masks])
        hf.create_dataset('images', data=all_images_np)
        hf.create_dataset('masks', data=all_masks_np)


def augment_image(images,angles=[0, 30, 60, 90, 120, 150, 180]):
    aug_images = []
    for img in images:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        for angle in angles:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_rotated = cv2.warpAffine(img, M, (w, h),borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            aug_images.append(img_rotated)
    return aug_images


def show_images(images, masks):
    for i in range(len(images)):
        images[i] = images[i].permute(1, 2, 0).numpy()
        masks[i] = masks[i].squeeze().numpy()

        cv2.imshow('image', images[i])
        cv2.imshow('mask', masks[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
 
    seeding(42)

    CONFIG = {
        'SIZE': (512, 512),
        'K_FOLD':5,
        'BATCH_SIZE': 4,
        'EPOCHS': 300,
        'LEARNING_RATE': 1e-3,  
        'PATIENCE': 50,
        'WEIGHT_DECAY':1e-5,
        'MOMENTUM': 0.999,
        
        'TRAIN_IMG_PATH':  "/unet/images/*",
        'TRAIN_MASK_PATH': "/unet/labels/*",


        'CHECKPOINT_PATH': "/checkpoints/",
        'TRAIN_PREPROCESSED_h5_PATH': "/preprocessed_data.h5",
    }



    ## CREATE IMAGES/MASKS ###
    # load the images/masks
    images = load_images(sorted(glob(CONFIG['TRAIN_IMG_PATH'])),type="image")
    masks = load_images(sorted(glob(CONFIG['TRAIN_MASK_PATH'])),type="mask")

    # augment the images/masks if needed
    images = augment_image(images)
    masks = augment_image(masks)

    # preprocess the images/masks
    images = preprocess(images,type="image")
    masks = preprocess(masks,type="mask")
    
    # show the images/masks
    show_images(images, masks)

    # save the preprocessed images/masks in h5 file so that we can use them later on different hyperparameter
    save_data(path=CONFIG["TRAIN_PREPROCESSED_h5_PATH"],all_images=images, all_masks=masks)



    all_images_tensor, all_masks_tensor = load_data(path=CONFIG["TRAIN_PREPROCESSED_h5_PATH"])

    kfold = KFold(n_splits=CONFIG["K_FOLD"], shuffle=True, random_state=42)
    
    all_best_train_accs = []
    all_best_valid_accs = []
    all_best_valid_loss = []
    all_best_dices = []
    all_best_ious = []

    for fold, (train_idx, valid_idx) in enumerate(tqdm(kfold.split(all_images_tensor), total=CONFIG["K_FOLD"], desc="KFold Progress")):


        # initialize the model for each fold
        model = SAUNet(in_channels=3, num_classes=1)
        trainer = UNetTrainer(CONFIG, model=model)

        train_dataset = TensorDataset(all_images_tensor[train_idx], all_masks_tensor[train_idx])
        valid_dataset = TensorDataset(all_images_tensor[valid_idx], all_masks_tensor[valid_idx])

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True,  num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4)
        
        best_train_acc_for_fold, best_valid_acc_for_fold,best_valid_loss,best_dice, best_iou  = trainer.train(train_loader, valid_loader, fold)

        all_best_train_accs.append(best_train_acc_for_fold)
        all_best_valid_accs.append(best_valid_acc_for_fold)
        all_best_valid_loss.append(best_valid_loss)
        all_best_dices.append(best_dice)
        all_best_ious.append(best_iou)


        print(f"Fold: {fold+1}/5 | Best training accuracy: {best_train_acc_for_fold:.3f} | Best validation accuracy: {best_valid_acc_for_fold:.3f} | Best validation loss: {best_valid_loss:.3f} | Best Dice: {best_dice:.3f} | Best IoU: {best_iou:.3f}")


    avg_best_train_acc = sum(all_best_train_accs) / len(all_best_train_accs)
    avg_best_valid_acc = sum(all_best_valid_accs) / len(all_best_valid_accs)
    avg_best_valid_loss = sum(all_best_valid_loss) / len(all_best_valid_loss)
    avg_best_dice = sum(all_best_dices) / len(all_best_dices)
    avg_best_iou = sum(all_best_ious) / len(all_best_ious)

    std_best_train_acc = np.std(all_best_train_accs)
    std_best_valid_acc = np.std(all_best_valid_accs)
    std_best_valid_loss = np.std(all_best_valid_loss)
    std_best_dice = np.std(all_best_dices)
    std_best_iou = np.std(all_best_ious)


    print(f"Training accuracy across all folds: {avg_best_train_acc:.3f} ± {std_best_train_acc:.3f}")
    print(f"Validation accuracy across all folds: {avg_best_valid_acc:.3f} ± {std_best_valid_acc:.3f}")
    print(f"Validation loss across all folds: {avg_best_valid_loss:.3f} ± {std_best_valid_loss:.3f}")
    print(f"Validation Dice across all folds: {avg_best_dice:.3f} ± {std_best_dice:.3f}")
    print(f"Validation IoU across all folds: {avg_best_iou:.3f} ± {std_best_iou:.3f}")

if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
