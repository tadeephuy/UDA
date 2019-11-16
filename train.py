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
from augmentation import apply_augmentations


if __name__ == "__main__":        
#     multiprocessing.freeze_support()
    ## Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--uda', help='use UDA', default=True)
    parser.add_argument('--output_dir', help='path to output folder', default='/home/ted/Projects/UDA/checkpoints')
    parser.add_argument('--cuda', type=bool, help='use GPU is available', default=True)
    parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use') # TODO
    parser.add_argument('--labeled_dir', help='path to training labeled data csv', default='/home/ted/Projects/Chest%20X-Ray%20Images%20Classification/data/CheXpert-v1.0-small/split/train.csv')
    parser.add_argument('--validation_dir', help='path to validation labeled data csv', default='/home/ted/Projects/Chest%20X-Ray%20Images%20Classification/data/CheXpert-v1.0-small/split/dev.csv')
    parser.add_argument('--unlabeled_dir', help='path to unlabeled data csv', default='/home/ted/Downloads/NIH_sample/sample_labels.csv')
    parser.add_argument('--n_workers', type=int, help='number of workers', default=0)
    parser.add_argument('--bs', type=int, help='batch_size', default=3)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-3)
    parser.add_argument('--n_epoch', type=int, help='number of epoch', default=3)
    parser.add_argument('--weight_dir', default='')
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
    labeled_dataset = LabeledDataset(csv_dir=labeled_dir)
    labeled_dataloader = torch.utils.data.DataLoader(dataset=labeled_dataset, batch_size=batch_size, 
                                                    shuffle=True, num_workers=num_workers)
    validation_dir = opt.validation_dir
    validation_dataset = LabeledDataset(csv_dir=labeled_dir)
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, 
                                                        shuffle=False, num_workers=num_workers + 5)
    if uda:
        # augmentations = transforms.Compose([transforms.RandomAffine(degrees=(-15, 15), translate=[0.05, 0.05],
        #                                                             scale=(0.95, 1.05), fillcolor=128)])
        unlabeled_dir = opt.unlabeled_dir
        unlabeled_dataset = UnlabeledDataset(csv_dir=unlabeled_dir, augmentations=apply_augmentations)
        unlabeled_dataloader = torch.utils.data.DataLoader(dataset=unlabeled_dataset, batch_size=batch_size, 
                                                        shuffle=True, num_workers=num_workers)
    # Model
    import torchvision.models as models
    model = models.resnext101_32x8d(groups=1, num_classes=5, zero_init_residual=True)
    model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    if opt.weight_dir:
        model.load_state_dict(torch.load(opt.weight_dir))
        print('weight loaded')
    else:
        model.apply(weights_init)
    model = model.to(device)

    # Optimization
    cross_entropy = nn.BCEWithLogitsLoss().to(device) # supervised loss
    kl_divergence = nn.KLDivLoss(reduction='batchmean').to(device) # unsupervised loss (consistency loss)

    if uda:
        supervised_weight, unsupervised_weight = 1.0, 5.0 
    else:
        supervised_weight, unsupervised_weight = 1.0, 0.0

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.99, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2)

    ## Training Loop
    ################
    writer = SummaryWriter()

    n_epoch = opt.n_epoch
    min_train_loss = 1.0
    min_val_loss = 1.0
    total_train_loss = 0.0
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
            img, label = img.to(device), label.to(device)
            sup_output = model(img)
            supervised_loss = cross_entropy(sup_output, label)
            supervised_loss *= supervised_weight
            
            train_loss = supervised_loss
            # Unsupervised branch #
            #######################
            if uda:
                ## sample from the Unlabeled Dataset
                unlabeled_img, noised_unlabeled_img = iter(unlabeled_dataloader).next()
                unlabeled_img, noised_unlabeled_img = unlabeled_img.to(device), noised_unlabeled_img.to(device)

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
                writer.add_scalars("Loss_Train", {
                    'L_sup': supervised_loss.item(),
                    'L_unsup': unsupervised_loss.item()
                }, epoch * len(labeled_dataloader) + i)

            total_train_loss += train_loss

            train_loss.backward()
            optimizer.step()

            e_t = time.time()
            n_iters_per_second = int(1 // (e_t - s_t))
            print(f'[{i}/{len(labeled_dataloader)}] L_train: {train_loss.item():.4f} n_it/s: {n_iters_per_second}')
            writer.add_scalar('L_train', train_loss.item(), epoch * len(labeled_dataloader) + i)

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
                        img, label = img.to(device), label.to(device)
                        output = model(img)
                        val_loss += cross_entropy(output, label)
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
                img, label = img.to(device), label.to(device)
                output = model(img)
                val_loss += cross_entropy(output, label)
            val_loss = val_loss / i

        # update learning rate base on val_loss
        scheduler.step(val_loss)
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

