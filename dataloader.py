import numpy as numpy
import panda as pd
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset

class LabeledDataset(Dataset):
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        self.data = pd.read_csv(self.csv_dir)
        self.label_list = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices"
        ]
        self.labels = self.data[self.label_list].to_numpy()

    def __getitem__(self, idx):
        sample_data = self.iloc[idx]
        img_dir = sample_data.Path
        img = cv2.imread(img_dir, 0) # grayscale

        label = self.labels[idx]
        return img, label

    def len(self):
        return len(self.data)


class UnlabeledDataset(Dataset):
    def __init__(self, csv_dir, augmentations):
        self.csv_dir = csv_dir
        self.data = pd.read_csv(self.csv_dir)
        self.augmentations = augmentations

    def __getitem__(self, idx):
        sample_data = self.iloc[idx]
        img_dir = sample_data.Path
        img = cv2.imread(img_dir, 0) # grayscale

        augmented_img = self.apply_augmentation(img)
        return img, augmented_img

    def len(self):
        return len(self.data)

    def apply_augmentation(self, img):
        return self.augmentations(img.copy())

        