import multiprocessing
import time
import argparse
# multiprocessing.set_start_method('spawn', True)
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from dataloader import LabeledDataset, UnlabeledDataset
from utils import weights_init, DiceLoss, AUCLoss, PropensityLoss, create_figure, calculate_auc
from augmentation import apply_augmentations_sup, augmentations_unsup
# print = lambda x: None

if __name__ == "__main__":        
#     multiprocessing.freeze_support()
    ## Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--uda', help='use UDA', action='store_true')
    parser.add_argument('--output_dir', help='path to output folder', default='/home/ted/Projects/UDA/checkpoints_SEDensenet121')
    parser.add_argument('--cuda', type=bool, help='use GPU is available', default=True)
    parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use') # TODO
    parser.add_argument('--labeled_dir', help='path to training labeled data csv', default='/home/ted/Projects/Chest%20X-Ray%20Images%20Classification/data/CheXpert-v1.0-small/split/train.csv')
    parser.add_argument('--validation_dir', help='path to validation labeled data csv', default='/home/ted/Projects/Chest%20X-Ray%20Images%20Classification/data/CheXpert-v1.0-small/split/dev.csv')
    parser.add_argument('--unlabeled_dir', help='path to unlabeled data csv', default='/vinbrain/data/NIH/imgs/Data_Entry_2017.csv')
    parser.add_argument('--n_workers', type=int, help='number of workers', default=0)
    parser.add_argument('--bs', type=int, help='batch_size', default=64)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--n_epoch', type=int, help='number of epoch', default=15)
    parser.add_argument('--weight_dir', default='')
    parser.add_argument('--beast', help='load dataset to RAM', action='store_true')
    opt = parser.parse_args()
    print(opt)
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
    
    labeled_dataset = LabeledDataset(csv_dir=labeled_dir, beast=opt.beast, n_samples=190000, augmentations=apply_augmentations_sup, prefix='/home/ted/Projects/Chest%20X-Ray%20Images%20Classification/data/')
    labeled_dataloader = torch.utils.data.DataLoader(dataset=labeled_dataset, batch_size=batch_size, 
                                                    shuffle=True, num_workers=num_workers)
    validation_dir = opt.validation_dir
    validation_dataset = LabeledDataset(csv_dir=validation_dir, n_samples=200, prefix='/home/ted/Projects/Chest%20X-Ray%20Images%20Classification/data/')
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, 
                                                        shuffle=False, num_workers=num_workers)
    
    test_dir = '/home/ted/Downloads/CheXpert-v1.0-small/valid.csv'
    test_dataset = LabeledDataset(csv_dir=test_dir, prefix='/home/ted/Projects/Chest%20X-Ray%20Images%20Classification/data/')
    test_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, 
                                                        shuffle=False, num_workers=num_workers)
    if uda:
        # augmentations = transforms.Compose([transforms.RandomAffine(degrees=(-15, 15), translate=[0.05, 0.05],
        #                                                             scale=(0.95, 1.05), fillcolor=128)])
        unlabeled_dir = opt.unlabeled_dir
        unlabeled_dataset = UnlabeledDataset(csv_dir=unlabeled_dir, augmentations=apply_augmentations_unsup, n_samples=100000, beast=False)
        unlabeled_dataloader = torch.utils.data.DataLoader(dataset=unlabeled_dataset, batch_size=batch_size, 
                                                        shuffle=True, num_workers=num_workers)
    # Model
    #### Resnet 152 ####
    # import torchvision.models as models
    # model = models.resnet152(num_classes=5)
    # model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    # if opt.weight_dir:
    #     model.load_state_dict(torch.load(opt.weight_dir))
    #     print('weight loaded')
    # else:
    #     model.apply(weights_init)
    ####################

    #### SE-Densenet 121 ####
    from densenet import se_densenet121
    model = se_densenet121(pretrained=False, num_channels=1, num_classes=5)
    if opt.weight_dir:
        model.load_state_dict(torch.load(opt.weight_dir))
        print('weight loaded')
    #########################
    model = model.to(device)

    # Optimization
    cross_entropy = nn.BCEWithLogitsLoss().to(device) # supervised loss
    dice_loss = DiceLoss().to(device)
    auc_loss = AUCLoss().to(device)
    kl_divergence = nn.KLDivLoss(reduction='batchmean').to(device) # unsupervised loss (consistency loss)
    labels_array = torch.from_numpy(labeled_dataset.labels).to(device)
    propensity_loss = PropensityLoss(labels_array).to(device) # take in labels set to calculate propensities
    # proxy_w*(ce_w*ce_loss + d_w*d_loss) + a_w*a_loss
    proxy_weight, ce_weight, d_weight, a_weight, p_weight = 1.0, 0.8, 0.0, 0.5, 1.5
    if uda:
        supervised_weight, unsupervised_weight = 1.0, 5.0
    else:
        supervised_weight, unsupervised_weight = 1.0, 0.0

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.99, nesterov=True, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(labeled_dataloader), 2, 1e-9)
    # scheduler = optim.lr_scheduler.CyclicLR()
    

    ## Training Loop
    ################
    writer = SummaryWriter()
    n_epoch = opt.n_epoch
    min_train_loss = 1.0
    min_val_loss = 1.0
    total_train_loss = 0.0
    max_auc = 0.0
    print("Start training")
    for epoch in range(n_epoch):
        s_t = time.time()
        ## Training
        model.train()
        for i, (img, label) in enumerate(labeled_dataloader):
            # if i > 30:
            #     break
            s_t = time.time()

            model.zero_grad()
            # Supervised branch #
            #####################
            img, label = img.float().to(device), label.float().to(device)
            if i % 100 == 0:
                fig = create_figure(img, label)
                writer.add_figure("Labeled", fig, epoch * len(labeled_dataloader) + i)
            sup_output = model(img)
            ce_loss, d_loss, a_loss = cross_entropy(sup_output, label), dice_loss(sup_output, label), auc_loss(sup_output, label)
            p_loss = propensity_loss(sup_output, label)
            supervised_loss = proxy_weight*(ce_weight*ce_loss + d_weight*d_loss) + a_weight*a_loss + p_weight*p_loss
            supervised_loss *= supervised_weight
            
            train_loss = supervised_loss
            if i % 20 == 0:
                writer.add_scalars("Supervised_Loss", {
                        'BCE': ce_loss.item(),
                        'Dice': d_loss.item(),
                        'AUC': a_loss.item(),
                        'Propensity': p_loss.item(),
                    }, epoch * len(labeled_dataloader) + i)
            # Unsupervised branch #
            #######################
            if uda:
                ## sample from the Unlabeled Dataset
                unlabeled_img, noised_unlabeled_img = iter(unlabeled_dataloader).next()
                unlabeled_img, noised_unlabeled_img = unlabeled_img.float().to(device), noised_unlabeled_img.float().to(device)
                if i % 100 == 0:
                    fig = create_figure(unlabeled_img, noised_unlabeled_img, uda=uda)
                    writer.add_figure("Unlabeled", fig, epoch * len(labeled_dataloader) + i)
                ## calculate the outputs from the two versions
                # only a fix copy (no gradient flows) when passing original unlabeled img
                unsup_ori_output = model(unlabeled_img).detach()
                unsup_ori_output = torch.nn.functional.log_softmax(unsup_ori_output, 1)
                # no detach(), still requires gradients to flow through this pass 
                unsup_aug_output = model(noised_unlabeled_img)
                unsup_aug_output = torch.nn.functional.softmax(unsup_aug_output, 1)

                ## loss calculation between 2 output probability distributions
                unsupervised_loss = kl_divergence(unsup_ori_output, unsup_aug_output)
                unsupervised_loss *= unsupervised_weight

                # TOTAL LOSS #
                train_loss += unsupervised_loss
                if i % 50 == 0:
                    writer.add_scalars("Loss_Train", {
                        'L_sup': supervised_loss.item(),
                        'L_unsup': unsupervised_loss.item()
                    }, epoch * len(labeled_dataloader) + i)

            total_train_loss += train_loss

            train_loss.backward()
            optimizer.step()
            scheduler.step()

            auc, _ = calculate_auc(sup_output.detach().cpu(), label.detach().cpu())
            e_t = time.time()
            n_seconds_per_iter = e_t - s_t
            if i % 50 == 0:
                print(f'[{epoch} - {i}/{len(labeled_dataloader)}] L_train: {train_loss.item():.4f} ROC: {auc:.4f} bestROC: {max_auc:.4f} | duration: {n_seconds_per_iter:.2f} s/it')
                writer.add_scalar('L_train', train_loss.item(), epoch * len(labeled_dataloader) + i)
                if auc != -1:
                    writer.add_scalar('AUC', auc, epoch * len(labeled_dataloader) + i)

            # start saving best model (train_loss) after 1000th iterations
            if (i > 500 or epoch > 0) and train_loss <= min_train_loss:
                min_train_loss = train_loss
                torch.save(model.state_dict(), f'{output_dir}/model_best_train.pth')
        
            # evaluation and save best model (val_loss) after every 300 iterations
            if i % 500 == 0 and i > 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for i_v, (img, label) in enumerate(validation_dataloader):
                        if i_v > 50:
                            break
                        img, label = img.float().to(device), label.float().to(device)
                        output = model(img)
                        val_loss += proxy_weight*(ce_weight*cross_entropy(output, label) + d_weight*dice_loss(output, label)) + a_weight*auc_loss(output, label)
                        print(f'L_val: {val_loss.item() / (i_v + 1):.4f}')
                    val_loss = val_loss / i_v
                    if val_loss <= min_val_loss:
                        min_val_loss = val_loss
                        torch.save(model.state_dict(), f'{output_dir}/model_best_val.pth')
                    writer.add_scalar('L_val', val_loss.item(), epoch * len(labeled_dataloader) + i)
                model.train()
                

        ## Evaluation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for i, (img, label) in enumerate(validation_dataloader):
                if i > 300:
                    break
                img, label = img.float().to(device), label.float().to(device)
                output = model(img)
                val_loss += proxy_weight*(ce_weight*cross_entropy(output, label) + d_weight*dice_loss(output, label)) + a_weight*auc_loss(output, label)
            val_loss = val_loss / i

        #### ON 200 Chexpert test set ####
        outputs = np.array([[0., 0., 0., 0., 0.]])
        labels = np.array([[0., 0., 0., 0., 0.]])
        with torch.no_grad():
            for img, label in test_dataloader:
                img, label = img.float().cuda(), label.float().cuda()
                output = model(img)
                outputs = np.vstack((outputs, output.detach().cpu().numpy()))
                labels = np.vstack((labels, label.detach().cpu().numpy()))
        auc, auc_binarized = calculate_auc(outputs[1:], labels[1:], 0.5)
        if auc >= max_auc:
            max_auc = auc
            torch.save(model.state_dict(), f'{output_dir}/model_best_auc.pth')
        writer.add_scalar('Test AUC', auc, epoch)
        ###################################

        # update learning rate base on val_loss
        # scheduler.step(val_loss)
        # save model every epoch
        torch.save(model.state_dict(), f'{output_dir}/model_{epoch}_{train_loss.item():.4f}_{val_loss.item():.4f}.pth')

        e_t = time.time()
        epoch_duration = e_t - s_t
        avg_train_loss = total_train_loss / len(labeled_dataloader)
        if uda:
            print(f'{epoch}/{n_epoch}: L_sup: {supervised_loss.item()} L_unsup: {unsupervised_loss.item()} L_train: {avg_train_loss.item()} L_val: {val_loss.item()} duration: {epoch_duration:.2f}s')
        else:
            print(f'{epoch}/{n_epoch}: L_sup: {supervised_loss.item()} L_train: {avg_train_loss.item()} L_val: {val_loss.item()} duration: {epoch_duration:.2f}s')
        writer.add_scalars('Epoch_Losses', {
            'Train': avg_train_loss.item(),
            'Val': val_loss.item()
        }, epoch)

    writer.close()

