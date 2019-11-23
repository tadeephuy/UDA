from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn as nn
from dataloader import LabeledDataset
from utils import calculate_auc

from tta import TestTimeAugmentation
from augmentation import apply_augmentations_tta

if __name__ == "__main__":
    is_tta = True
    test_dataset = LabeledDataset("/home/ted/Projects/Chest%20X-Ray%20Images%20Classification/data/CheXpert-v1.0-small/valid.csv", 
                                prefix='/home/ted/Projects/Chest%20X-Ray%20Images%20Classification/data/', raw=is_tta)
    dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1)

    import torchvision.models as models
    # model = models.resnet152(num_classes=5)
    # model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    # model.load_state_dict(torch.load("/home/ted/Projects/no_model_98_0.2959_0.4846.pth"))
    from densenet import se_densenet121
    model = se_densenet121(pretrained=False, num_channels=1, num_classes=5)
    model.load_state_dict(torch.load("/home/ted/no_se_model_best_auc_21.pth"))
    model.eval()
    model.cuda()
    tta = TestTimeAugmentation(model, apply_augmentations_tta, 5)
    outputs = np.array([[0., 0., 0., 0., 0.]])
    labels = np.array([[0., 0., 0., 0., 0.]])
    with torch.no_grad():
        for img, label in dataloader:
            if is_tta:
                img = np.squeeze(img, 0).cpu().numpy()
                output = tta.predict(img)
            else:
                img, label = img.float().cuda(), label.float().cuda()
                output = model(img)
            output = torch.sigmoid(output)
            outputs = np.vstack((outputs, output.detach().cpu().numpy()))
            labels = np.vstack((labels, label.detach().cpu().numpy()))
    auc, auc_binarized = calculate_auc(outputs[1:], labels[1:], 0.5)
    print(auc, auc_binarized)
    
