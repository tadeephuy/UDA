import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from utils import img_to_tensor


class TestTimeAugmentation():
    def __init__(self, model, augmentations=None, n_predictions=5, ensemble_mode='mean', img_size=None):
        super(TestTimeAugmentation, self).__init__()
        self.model = model
        self.augmentations = None
        self.n_predictions = n_predictions
        self.ensemble_mode = ensemble_mode
        self.img_size = img_size

        self.set_augmentations(augmentations)

    def predict(self, x):
        ## preprocess image
        x = self._preprocess(x)
        ## augment images
        augmented_imgs = self._augment_imgs(x) # shape NxWxH
        ## predictions
        preds = self.model(augmented_imgs) # batch prediction
        ## ensemble
        ensembled_pred = self._ensemble_pred(preds)
        return ensembled_pred

    def set_augmentations(self, augmentations):
        self.augmentations = augmentations
    
    def _preprocess(self, x):
        if self.img_size:
            x = cv2.resize(x, self.img_size)
        return x

    def _augment_imgs(self, x):
        augmented_imgs = []
        for _ in range(self.n_predictions):
            augmented_img = img_to_tensor(self.augmentations(x)).float().cuda()
            augmented_img.unsqueeze_(0)
            augmented_imgs.append(augmented_img)
        augmented_imgs = torch.cat(augmented_imgs)
        return augmented_imgs

    def _ensemble_pred(self, preds):
        ensembled_pred = torch.mean(preds, dim=0)
        return ensembled_pred

if __name__ == '__main__':
    from densenet import se_densenet121
    from augmentation import apply_augmentations_tta

    model = se_densenet121(pretrained=False, num_channels=1, num_classes=5)
    model.load_state_dict(torch.load('/home/ted/no_se_model_best_auc_21.pth'))
    model.eval()

    ##
    img = cv2.imread("/home/ted/Downloads/CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg", 0)
    ##
    with torch.no_grad():
        tta = TestTimeAugmentation(model, augmentations=apply_augmentations_tta)
        pred = tta.predict(img)
    # print(pred)