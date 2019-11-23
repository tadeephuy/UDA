from torch import nn
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def img_to_tensor(img_array):
    """
    take a np img array and convert to torch tensor
    """
    img = Image.fromarray(img_array, 'L')
    img = transforms.ToTensor()(img)
    return img

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self,pred, target):
       smooth = 1e-6
       iflat = pred.contiguous().view(-1)
       tflat = target.contiguous().view(-1)
       intersection = (iflat * tflat).sum()
       A_sum = torch.sum(iflat * iflat)
       B_sum = torch.sum(tflat * tflat)
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

class AUCLoss(torch.nn.Module):
    def __init__(self):
        super(AUCLoss, self).__init__()
    def forward(self, pred, target):
        a = torch.sigmoid(torch.mm(pred, pred.T))
        b = torch.mm(target, torch.ones(target.shape).T.cuda())
        c = torch.mm(torch.ones(target.shape).cuda(), target.T)
        cost = torch.mean(a * torch.from_numpy(np.maximum(b.cpu().numpy() - c.cpu().numpy(), 0)).cuda())
        return 1 - cost

class PropensityLoss(torch.nn.Module):
    """
    Implementation based on
    'Extreme Multi-label Loss Functions for 
    Recommendation, Tagging, Ranking & Other Missing Label Applications'
    http://ceur-ws.org/Vol-2126/paper10.pdf
    """
    def __init__(self, labels_array):
        super(PropensityLoss, self).__init__()
        propensities = self.calculate_propensities(labels_array)
        self.propensities = propensities

    def forward(self, pred, target):
        N = pred.shape[0]
        pred = torch.sigmoid(pred).cuda()
        cost = torch.sum((1 / self.propensities) * (2*pred - 1) * ((target - pred) ** 2)) / N
        return 1 - cost

    @staticmethod
    def calculate_propensities(labels_array):
        """
        labels_array of size NxNl where N is the number of samples 
        and Nl is the total counts of each class within the dataset
        """
        N = torch.Tensor([labels_array.shape[0]]).cuda()
        Nl = torch.sum(labels_array, dim=0).cuda()
        propensities = 1 / (1 + (torch.log(N - 1)).double() * 1.183 * torch.exp(-0.5 * torch.log(Nl + 0.4).double()).double())
        return propensities

    def _augment_imgs(self, x):
        augmented_imgs = []
        for _ in range(self.n_predictions):
            augmented_img = img_to_tensor(self.augmentations(x))
            augmented_imgs.append(augmented_img)
        augmented_imgs = torch.cat(augmented_imgs)
        return augmented_imgs

    def _ensemble_pred(self, preds):
        ensembled_pred = torch.mean(preds, dim=0)
        return ensembled_pred

def create_figure(img, labels, uda=False):
    if uda:
        # concat images
        batch_size = img.shape[0]
        in_img, labels = img.clone().detach().cpu(), labels.clone().detach().cpu()
        grid_ori = np.transpose(torchvision.utils.make_grid(in_img, batch_size, pad_value=1), (1, 2, 0))
        grid_noise = np.transpose(torchvision.utils.make_grid(labels, batch_size, pad_value=1), (1, 2, 0))
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 5), gridspec_kw={'hspace': 0.6 })
        fig.suptitle('Images')
        ax1.imshow(grid_ori)
        ax1.set_title("Original")
        ax2.imshow(grid_noise)
        ax2.set_title("Noised")
    else:
        # concat images
        batch_size = img.shape[0]
        in_img = img.clone().detach().cpu()
        grid_in = np.transpose(torchvision.utils.make_grid(in_img, batch_size, pad_value=1), (1, 2, 0))
        fig, ax1 = plt.subplots(1, figsize=(10, 5), gridspec_kw={'hspace': 0.6 })
        fig.suptitle('Images')
        ax1.imshow(grid_in)
        ax1.set_title(str(labels) + " SIZE " + str(in_img[0].shape))
    return fig

def calculate_auc(output, label, threshold=0.6):
    """
    output, label shape: B x n_class
    """
    binarize = np.vectorize(lambda x: 1.0 if x >= threshold else 0.0)

    binarized_output = binarize(output)
    try:
        auc = roc_auc_score(label, output)
        auc_binarized = roc_auc_score(label, binarized_output)
    except ValueError:
        return -1.0, -1.0
    return auc, auc_binarized


