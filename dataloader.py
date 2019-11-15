import numpy as numpy
import pandas as pd
from PIL import Image
import cv2
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

class LabeledDataset(Dataset):
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        self.data = pd.read_csv(self.csv_dir)
        self.data.fillna(0)
        label_names_0 = ['Cardiomegaly', 'Consolidation',
                         'No Finding', 'Enlarged Cardiomediastinum', 'Pneumonia', 'Pneumothorax', 'Pleural Other']
        self.data[label_names_0] = 1 * (self.data[label_names_0] > 0)  # convert uncertain -1 to negative 0
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
        self.labels = self.data[self.label_list]
        self.labels = self.labels.apply(lambda x: abs(x)).to_numpy()

    def __getitem__(self, idx):
        sample_data = self.data.iloc[idx]
        img_dir = "/home/ted/Projects/Chest%20X-Ray%20Images%20Classification/data/" + sample_data.Path
        img = cv2.imread(img_dir, 0) # grayscale
        img = Image.fromarray(img, 'L')
        img = transforms.Compose([transforms.Resize((320, 320)), transforms.ToTensor()])(img)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.data)


class UnlabeledDataset(Dataset):
    def __init__(self, csv_dir, augmentations):
        self.csv_dir = csv_dir
        self.data = pd.read_csv(self.csv_dir)
        self.augmentations = augmentations

    def __getitem__(self, idx):
        sample_data = self.data.iloc[idx]
        img_dir = sample_data.Path
        img = cv2.imread(img_dir, 0) # grayscale

        augmented_img = self.apply_augmentation(img)
        return img, augmented_img

    def __len__(self):
        return len(self.data)

    def apply_augmentation(self, img):
        return self.augmentations(img.copy())

        