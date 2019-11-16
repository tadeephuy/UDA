from torch import nn
from PIL import Image
from torchvision import transforms

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def img_to_tensor(img_array):
    img = Image.fromarray(img_array, 'L')
    img = transforms.ToTensor()(img)
    return img