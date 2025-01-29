from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks
        self.n_samples = len(images)

    def __getitem__(self, index):
        return self.images[index], self.masks[index]

    def __len__(self):
        return self.n_samples
