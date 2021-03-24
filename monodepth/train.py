import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
from tensorboardX import SummaryWriter

from model import PTModel, PTResModel
from loss import ssim
from data import *
from utils import *
import numpy as np
from pairs_gen import AllPositivePairSelector
from losses import OnlineContrastiveLoss


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--data-folder', type=str, help='location of dataset folder')
    parser.add_argument('--pretrained-weights', type=str, default=None, help='location of pretrained weights')
    parser.add_argument('--label-file', type=str, help='name of label txt file')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    parser.add_argument('--num-instances', default=None, type=int, help='number of instances of same image in the batch')
    parser.add_argument('--lamb', default=0., type=float,
                    help='lambda: multiplier for added additional loss')
    parser.add_argument('--evaluate', type=str, default=None, help='path of model file for testing')
    args = parser.parse_args()
    print(args)
    # Create model
    model = PTResModel(pretrained_weights=args.pretrained_weights)
    evaluating=(args.evaluate!=None)
    if(evaluating):
        print('Evaluating ', args.evaluate)
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint)

    model = model.cuda()
    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam( model.parameters(), args.lr )
    batch_size = args.bs

    # Load data

    data_test, nyu2_test = loadToMem(args.data_folder, txtfile='/nyu2_test_updated.csv')
    transformed_testing = depthDatasetMemory(data_test, nyu2_test, transform=getNoTransform(is_test=True))
    test_loader = DataLoader(transformed_testing, 1, shuffle=False)
    if(evaluating):
        evaluate_model(model, test_loader, args)
        return

    data, nyu2_train = loadToMem(args.data_folder, txtfile=args.label_file)
    transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
    if(args.num_instances):
        train_loader = DataLoader(transformed_training, batch_size, sampler=RandomIdentitySampler(transformed_training, num_instances=args.num_instances), num_workers=4, drop_last=True)
    else:
        train_loader = DataLoader(transformed_training, batch_size, shuffle=True, num_workers=4, drop_last=True)


    # Logging
    writer = SummaryWriter('logs', flush_secs=30)

    # Loss
    l1_criterion = nn.L1Loss()
    l1_criterion = l1_criterion.cuda()
    ps = AllPositivePairSelector(balance=False)
    criterion2 = OnlineContrastiveLoss(1., ps)
    criterion2 = criterion2.cuda()
    # Start training...
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)

        # Switch to train mode
        model.train()
        transformed_training.reset_seeds()
        end = time.time()

        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
            image_id = torch.autograd.Variable(sample_batched['image_id'].cuda(non_blocking=True))
            # Normalize depth
            depth_n = DepthNorm( depth )

            # Predict
            output, feats = model(image)

            # Compute the loss
            l_depth = l1_criterion(output, depth_n)
            l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
            
            loss = (1.0 * l_ssim) + (0.1 * l_depth)
            
            if(args.lamb>0):
                l_mse = criterion2(feats, image_id, neg_loss=False)
                loss += args.lamb * l_mse

            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
        
            # Log progress
            niter = epoch*N+i
            if i % 5 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

                # Log to tensorboard
                writer.add_scalar('Train/Loss', losses.val, niter)

            if i % 300 == 0:
                LogProgress(model, writer, test_loader, niter)

        # Record epoch's intermediate results
        LogProgress(model, writer, test_loader, niter)
        writer.add_scalar('Train/Loss.avg', losses.avg, epoch)
        torch.save(model.state_dict(), "models_"+str(epoch)+'.pth.tar')
        evaluate_model(model, test_loader, args)


def LogProgress(model, writer, test_loader, epoch):
    model.eval()
    sequential = test_loader
    sample_batched = next(iter(sequential))
    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
    if epoch == 0: writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
    if epoch == 0: writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), epoch)
    output = DepthNorm( model(image) )
    writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
    writer.add_image('Train.3.Diff', colorize(vutils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False)), epoch)
    del image
    del depth
    del output
    model.train()

if __name__ == '__main__':
    main()
