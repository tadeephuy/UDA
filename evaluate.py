from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn as nn
from dataloader import LabeledDataset
from utils import calculate_auc

if __name__ == "__main__":
    test_dataset = LabeledDataset("/home/ted/Projects/Chest%20X-Ray%20Images%20Classification/data/CheXpert-v1.0-small/valid.csv", 
                                prefix='/home/ted/Projects/Chest%20X-Ray%20Images%20Classification/data/')
    dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=4)

    import torchvision.models as models
    # model = models.resnext101_32x8d(groups=1, num_classes=5, zero_init_residual=True)
    model = models.resnet152(num_classes=5)
    model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    model.load_state_dict(torch.load("/home/ted/Projects/UDA/checkpoints/model_best_train.pth"))
    model.cuda()
    outputs = np.array([[0., 0., 0., 0., 0.]])
    labels = np.array([[0., 0., 0., 0., 0.]])
    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.float().cuda(), label.float().cuda()
            output = model(img)
            outputs = np.vstack((outputs, output.detach().cpu().numpy()))
            labels = np.vstack((labels, label.detach().cpu().numpy()))
    auc, auc_binarized = calculate_auc(outputs[1:], labels[1:], 0.5)
    print(auc, auc_binarized)
    
