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
    def init(self):
        super(DiceLoss, self).init()
    def forward(self,pred, target):
       smooth = 1.
       iflat = pred.contiguous().view(-1)
       tflat = target.contiguous().view(-1)
       intersection = (iflat * tflat).sum()
       A_sum = torch.sum(iflat * iflat)
       B_sum = torch.sum(tflat * tflat)
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

class AUCLoss(torch.nn.Module):
    def init(self):
        super(AUCLoss, self).init()
    def forward(self, pred, target):
        a = torch.sigmoid(torch.mm(pred, pred.T))
        b = torch.mm(target, torch.ones(target.shape).T.cuda())
        c = torch.mm(torch.ones(target.shape).cuda(), target.T)
        cost = torch.mean(a * torch.from_numpy(np.maximum(b.cpu().numpy() - c.cpu().numpy(), 0)).cuda())
        return 1 - cost

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


