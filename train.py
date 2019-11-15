import multiprocessing
import time
import argparse
# multiprocessing.set_start_method('spawn', True)
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from dataloader import LabeledDataset, UnlabeledDataset
from utils import weights_init

## Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--uda', type=bool, help='use UDA', default=False)
parser.add_argument('--output_dir', help='path to output folder', default='/home/ted/Projects/UDA/checkpoints')
parser.add_argument('--cuda', type=bool, help='use GPU is available', default=True)
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use') # TODO
parser.add_argument('--labeled_dir', help='path to training labeled data csv', default='/home/ted/Projects/Chest%20X-Ray%20Images%20Classification/data/CheXpert-v1.0-small/split/train.csv')
parser.add_argument('--validation_dir', help='path to validation labeled data csv', default='/home/ted/Projects/Chest%20X-Ray%20Images%20Classification/data/CheXpert-v1.0-small/split/dev.csv')
parser.add_argument('--unlabeled_dir', help='path to unlabeled data csv', default='')
parser.add_argument('--n_workers', type=int, help='number of workers', default=10)
parser.add_argument('--bs', type=int, help='batch_size', default=8)
parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--n_epoch', type=int, help='number of epoch', default=5)
opt = parser.parse_args()
print(opt)


if __name__ == "__main__":        
#     multiprocessing.freeze_support()

    cudnn.benchmark = True # set to False if input sizes vary alot
    
    if opt.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    uda = opt.uda
    output_dir = opt.output_dir
    ## Initialization
    #################


    # Dataloader

    labeled_dir = opt.labeled_dir
    batch_size = opt.bs
    num_workers = opt.n_workers
    labeled_dataset = LabeledDataset(csv_dir=labeled_dir)
    labeled_dataloader = torch.utils.data.DataLoader(dataset=labeled_dataset, batch_size=batch_size, 
                                                    shuffle=True, num_workers=num_workers)
    validation_dir = opt.validation_dir
    validation_dataset = LabeledDataset(csv_dir=labeled_dir)
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, 
                                                        shuffle=False, num_workers=num_workers + 5)
    if uda:
        unlabeled_dir = opt.unlabeled_dir
        unlabeled_dataset = UnlabeledDataset(csv_dir=unlabeled_dir, augmentations=augmentations)
        unlabeled_dataloader = torch.utils.data.DataLoader(dataset=unlabeled_dataset, batch_size=batch_size, 
                                                        shuffle=True, num_workers=num_workers)
    # Model
    import torchvision.models as models
    model = models.resnext101_32x8d(groups=1, num_classes=14, zero_init_residual=True)
    model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    model = model.to(device)
    model.apply(weights_init)

    # Optimization
    cross_entropy = nn.BCEWithLogitsLoss().to(device) # supervised loss
    kl_divergence = nn.KLDivLoss(reduction='batchmean').to(device) # unsupervised loss (consistency loss)

    if uda:
        supervised_weight, unsupervised_weight = 0.4, 0.6 
    else:
        supervised_weight, unsupervised_weight = 1.0, 0.0

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1)

    ## Training Loop
    ################
    writer = SummaryWriter()

    n_epoch = opt.n_epoch
    min_train_loss = 0.0
    min_val_loss = 0.0
    print("Start training")
    for epoch in range(n_epoch):
        ## Training
        model.train()
        for i, (img, label) in enumerate(labeled_dataloader):
            s_t = time.time()

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
                unsup_ori_output = torch.softmax(unsup_ori_output)
                # no detach(), still requires gradients to flow through this pass 
                unsup_aug_output = model(noised_unlabeled_img)
                unsup_aug_output = torch.log_softmax(unsup_aug_output)

                ## loss calculation between 2 output probability distributions
                unsupervised_loss = kl_divergence(unsup_ori_output, unsup_aug_output)

                # TOTAL LOSS #
                train_loss += unsupervised_weight*unsupervised_loss

            train_loss.backward()
            optimizer.step()

            e_t = time.time()
            n_iters_per_second = 1 // (e_t - s_t)
            print(f'[{i}/{len(labeled_dataloader)//batch_size}] L_train: {train_loss.item()} n_it/s: {n_iters_per_second}')

            # start saving best model (train_loss) after 1000th iterations
            if (i > 1000 or epoch > 0) and train_loss <= min_train_loss:
                min_train_loss = train_loss
                torch.save(model.state_dict(), f'{output_dir}/model_best_train.pth')
        
            # evaluation and save best model (val_loss) after every 300 iterations
            if i % 300 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for i, (img, label) in enumerate(validation_dataloader):
                        img, label = img.to(device), label.to(device)
                        output = model(img)
                        val_loss += cross_entropy(output, label)
                    val_loss = val_loss / i
                    if val_loss <= min_val_loss:
                        min_val_loss = val_loss
                        torch.save(model.state_dict(), f'{output_dir}/model_best_val.pth')
                model.train()
            

        ## Evaluation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for i, (img, label) in enumerate(validation_dataloader):
                img, label = img.to(device), label.to(device)
                output = model(img)
                val_loss += cross_entropy(output, label)
            val_loss = val_loss / i

        # update learning rate base on val_loss
        scheduler.step(val_loss)
        # save model every epoch
        torch.save(model.state_dict(), f'{output_dir}/model_{epoch}_{str(train_loss.item()):.4f}_{str(val_loss.item()):.4f}.pth')

        if uda:
            print(f'{epoch}/{n_epoch}: L_sup: {supervised_loss.item()} L_unsup: {unsupervised_loss.item()} L_train: {train_loss.item()} L_val: {val_loss.item()}')
        else:
            print(f'{epoch}/{n_epoch}: L_sup: {supervised_loss.item()} L_train: {train_loss.item()} L_val: {val_loss.item()}')


