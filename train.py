import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from dataloader import LabeledDataset, UnlabeledDataset
from utils import weights_init

cudnn.benchmark = True # set to False if input sizes vary alot
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

uda = True

## Initialization
#################

# Dataloader

labeled_dir = ''
batch_size = 16
num_workers = 20
labeled_dataset = LabeledDataset(csv_dir=labeled_dir)
labeled_dataloader = torch.utils.data.DataLoader(dataset=labeled_dataset, batch_size=batch_size, 
                                                 shuffle=True, num_workers=num_workers)
validation_dir = ''
validation_dataset = LabeledDataset(csv_dir=labeled_dir)
validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, 
                                                    shuffle=False, num_workers=num_workers)
if uda:
    unlabeled_dir = ''
    unlabeled_dataset = UnlabeledDataset(csv_dir=unlabeled_dir, augmentations=augmentations)
    unlabeled_dataloader = torch.utils.data.DataLoader(dataset=unlabeled_dataset, batch_size=batch_size, 
                                                       shuffle=True, num_workers=num_workers)
# Model
import torchvision.models as models
model = models.resnext101_32x8d(num_classes=4, zero_init_residual=True, 
                                replace_stride_with_dilation=True, norm_layer=True).to(device)
model.apply(weights_init)

# Optimization
cross_entropy = nn.CrossEntropyLoss().to(device) # supervised loss
kl_divergence = nn.KLDivLoss(reduction='batchmean').to(device) # unsupervised loss (consistency loss)
supervised_weight, unsupervised_weight = 0.4, 0.6 if uda else 1.0, 0

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1)
## Training Loop
################
n_epoch = 10
min_train_loss = 0.0
min_val_loss = 0.0
for epoch in range(n_epoch):
    model.train()
    for i, (img, label) in enumerate(labeled_dataloader):
        model.zero_grad()
        # Supervised branch #
        #####################
        img, label = img.to(device), label.to(device)
        sup_output = model(img)
        supervised_loss = cross_entropy(sup_output, label)
        
        train_loss = supervised_weight*supervised_loss
        # Unsupervised branch #
        #######################
        if uda:
            ## sample from the Unlabeled Dataset
            unlabeled_img, noised_unlabeled_img = iter(unlabeled_dataloader).next()
            unlabeled_img, noised_unlabeled_img = unlabeled_img.to(device), noised_unlabeled_img.to(device)

            ## calculate the outputs from the two versions
            # only a fix copy (no gradient flows) when passing original unlabeled img
            unsup_ori_output = model(unlabeled_img).detach()
            # no detach(), still requires gradients to flow through this pass 
            unsup_aug_output = model(noised_unlabeled_img)

            ## loss calculation between 2 output probability distributions
            unsupervised_loss = kl_divergence(unsup_ori_output, unsup_aug_output)

            # TOTAL LOSS #
            train_loss += unsupervised_weight*unsupervised_loss

        train_loss.backward()
        optimizer.step()

        if i > 1000 and train_loss <= min_train_loss:
            min_train_loss = train_loss
            torch.save(model.state_dict(), f'{output_dir}/model_best_train.pth')

    model.eval()
    with torch.no_grad():
        for i, (img, label) in enumerate(validation_dataloader):
            img, label = img.to(device), label.to(device)
            output = model(img)
            val_loss = cross_entropy(output, label)
        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'{output_dir}/model_best_val.pth')
    scheduler.step(val_loss)
    torch.save(model.state_dict(), f'{output_dir}/model_{epoch}_{str(train_loss.item()):.4f}_{str(val_loss.item()):.4f}.pth')


