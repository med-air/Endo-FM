import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from dataset import *
import medpy


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = medpy.metric.binary.dc(pred, gt)
        hd95 = medpy.metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1., 0.
    else:
        return 0., 0.


def eval_dice(pred_y, gt_y, classes=5):
    pred_y = torch.argmax(torch.softmax(pred_y, dim=1), dim=1)
    all_dice = []
    all_hd95 = []
    # print(pred_y.shape, gt_y.shape, classes)

    pred_y = pred_y.cpu().detach().numpy()
    gt_y = gt_y.cpu().detach().numpy()

    for cls in range(1, classes):
        dice, hd95 = calculate_metric_percase(pred_y == cls, gt_y == cls)
        # print(cls, dice, hd95)

        all_dice.append(dice)
        all_hd95.append(hd95)

    all_dice = torch.tensor(all_dice).cuda()
    all_hd95 = torch.tensor(all_hd95).cuda()
    # exit(0)

    return torch.mean(all_dice), torch.mean(all_hd95)


def eval(model, val_loader, device, classes):
    all_dice = []
    all_hd95 = []
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # img, label, name = batch['image'].to(device), batch['label'].to(device).squeeze(0), batch['case_name']
            img, label = batch['image'].to(device).squeeze(0), batch['label'].to(device).squeeze(0)
            output = model(img)

            # print(img.shape, label.shape, output.shape)
            # exit(0)

            # dice
            dice, hd95 = eval_dice(output, label, classes=classes)
            all_dice.append(dice.item())
            all_hd95.append(hd95.item())

    return np.mean(np.array(all_dice)), np.mean(np.array(all_hd95))


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    train_transform = transforms.Compose([
        # RandomRescale(0.5, 1.2),
        # RandomCrop((input_size, input_size)),
        # RandomColor(),
        Resize((args.img_size + 32, args.img_size + 32)),
        RandomCrop((args.img_size, args.img_size)),
        RandomFlip(),
        RandomRotation(),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5])
    ])
    test_transform = transforms.Compose([
        Resize((args.img_size, args.img_size)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5])
    ])

    db_train = SegCTDataset(dataroot=args.root_path, mode='train', transforms=train_transform)
    db_test = SegCTDataset(dataroot=args.root_path, mode='test', transforms=test_transform)

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of test set is: {}".format(len(db_test)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    if args.test:
        model.eval()
        model.load_state_dict(torch.load(args.pretrained_model_weights, map_location='cpu'))
        test_dice, test_hd95 = eval(model, testloader, 'cuda', classes=2)
        print('Test Dice: %.1f, HD95: %.1f' % (test_dice * 100., test_hd95))
        exit(0)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # print(model)

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5e-2)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_dice = 0.
    best_hd95 = 0.
    # iterator = tqdm(range(max_epoch), ncols=70)
    iterator = range(max_epoch)
    for epoch_num in iterator:
        model.train()

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'].squeeze(0), sampled_batch['label'].squeeze(0)
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            # print(image_batch.shape, label_batch.shape, outputs.shape)
            # print(torch.max(label_batch).item(), torch.min(label_batch).item(), torch.sum(label_batch).item())
            # print(torch.max(outputs).item(), torch.min(outputs).item(), torch.sum(outputs).item())
            # exit(0)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # if iter_num % 10 == 1:
            #     logging.info('iteration %d - loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                index = 0
                image = image_batch[index]
                image = (image - image.min()) / (image.max() - image.min())
                # print(image.shape)
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs[index], dim=0), dim=0, keepdim=False).unsqueeze(0)
                outputs = outputs.repeat(3, 1, 1)
                # print(outputs.shape)
                writer.add_image('train/Prediction', outputs * 255, iter_num)
                labs = label_batch[index]
                labs = labs.repeat(3, 1, 1)
                # print(labs.shape)
                writer.add_image('train/GroundTruth', labs * 255, iter_num)
                # exit(0)

        # train_dice, train_hd5 = eval(model, trainloader, 'cuda', classes=2)
        # print('Epoch [%3d/%3d], Train Dice: %.4f' % (epoch_num + 1, max_epoch, train_dice))
        test_dice, test_hd95 = eval(model, testloader, 'cuda', classes=2)
        if test_dice > best_dice:
            best_dice = test_dice
            best_hd95 = test_hd95
        print('Epoch [%3d/%3d], Loss: %.4f, Dice: %.1f, HD95: %.1f, Best Dice: %.1f, Best HD95: %.1f' %
              (epoch_num + 1, max_epoch, loss.item(), test_dice * 100., test_hd95, best_dice * 100., best_hd95))

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            # iterator.close()
            break

    writer.close()
    return "Training Finished!"
