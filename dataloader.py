import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from utils import img_to_tensor
class LabeledDataset(Dataset):
    def __init__(self, csv_dir, augmentations=None, n_samples=200, beast=False, uda=False, prefix='', raw=False):
        self.csv_dir = csv_dir
        self.data = pd.read_csv(self.csv_dir)
        self.data.fillna(0, inplace=True)
        label_names_0 = ['Cardiomegaly', 'Consolidation',
                         'No Finding', 'Enlarged Cardiomediastinum', 'Pneumonia', 'Pneumothorax', 'Pleural Other']
        self.data[label_names_0] = 1 * (self.data[label_names_0] > 0)  # convert uncertain -1 to negative 0
        self.label_list = [
            # "No Finding",
            # "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            # "Lung Opacity",
            # "Lung Lesion",
            "Edema",
            "Consolidation",
            # "Pneumonia",
            "Atelectasis",
            # "Pneumothorax",
            "Pleural Effusion",
            # "Pleural Other",
            # "Fracture",
            # "Support Devices"
        ]
        self.labels = self.data[self.label_list]
        self.labels = self.labels.apply(lambda x: abs(x)).to_numpy()
        self.prefix = prefix
        self.uda = uda
        self.augmentations = None
        if augmentations:
            self.augmentations = augmentations

        # BEAST mode that load whole dataset to RAM
        self.beast = beast
        if self.beast:
            self.imgs = []
            self.labels_ = []
            print('Loading whole dataset to RAM')
            for idx, sample_data in self.data.sample(n_samples).iterrows():
                img_dir = self.prefix + sample_data.Path
                img = cv2.imread(img_dir, 0) # grayscale
                self.labels_.append(self.labels[idx])
                self.imgs.append(img)
            self.labels = np.array(self.labels_)
        print(f"number of samples: {self.__len__()}")
        self.raw = raw

        

    def __getitem__(self, idx):
        if self.beast:
            img = self.imgs[idx].copy()
        else:
            sample_data = self.data.iloc[idx]
            img_dir = self.prefix + sample_data.Path
            img = cv2.imread(img_dir, 0) # grayscale
        img = cv2.resize(img, (320, 320))
        # apply augmentations if not in UDA mode
        if not self.uda and self.augmentations:
            img = self.augment(img)
        if not self.raw:
            img = img_to_tensor(img)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        if self. beast:
            return len(self.imgs)
        return len(self.data)

    def augment(self, img):
        img = self.augmentations(img)
        return img


class UnlabeledDataset(Dataset):
    def __init__(self, csv_dir, augmentations, n_samples=200, beast=False, prefix=''):
        self.csv_dir = csv_dir
        self.data = pd.read_csv(self.csv_dir)
        self.augmentations = augmentations
        self.beast = beast
        self.prefix = prefix
        if self.beast:
            self.imgs = []
            print('Loading whole unlabel dataset to RAM')
            for idx, sample_data in self.data.sample(n_samples).iterrows():
                img_dir = self.prefix + sample_data['Image Index']
                print(img_dir)
                img = cv2.imread(img_dir, 0) # grayscale
                img = cv2.resize(img, (320, 320))
                self.imgs.append(img)

    def __getitem__(self, idx):
        if self.beast:
            img = self.imgs[idx]
        else:
            sample_data = self.data.iloc[idx]
            img_dir = self.prefix + sample_data['Image Index']
            img = cv2.imread(img_dir, 0) # grayscale
            img = cv2.resize(img, (320, 320))
        
        augmented_img = self.augment(img.copy())

        img, augmented_img = img_to_tensor(img), img_to_tensor(augmented_img)
        return img, augmented_img

    def __len__(self):
        if self.beast:
            return len(self.imgs)
        return len(self.data)

    def augment(self, img):
        img = self.augmentations(img)
        return img