from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# code modified from pytorch imagenet example

import argparse
import os
import time
import math
import random
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from collections import OrderedDict

import torchvision

from torch.utils.tensorboard import SummaryWriter
from transformations import get_train_transforms, get_val_transforms
from transformations import get_cub_train_transforms, get_cub_val_transforms
from transformations import get_spad_train_transforms, get_spad_val_transforms
from transformations import get_imagenet_train_transforms, get_imagenet_val_transforms
from transformations import get_denoiser_train_transforms, get_denoiser_val_transforms
import torchvision.transforms as transforms
from custom_dataloader import IMMetricLoader
from utils import AverageMeter
from pairs_gen import AllPositivePairSelector
from collections import defaultdict
from losses import OnlineContrastiveLoss
from torchvision.models.resnet import BasicBlock, Bottleneck
import networks
from metric_sampler import RandomIdentitySampler
from unet import UNet
from denoiser import *

# the arg parser
parser = argparse.ArgumentParser(description='PyTorch Image Classification')
parser.add_argument('data_folder', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--eval-count', default=1, type=int,
                    help='number of times to evaluate during each epoch')
parser.add_argument('--warmup-epochs', default=0, type=int,
                    help='number of epochs for warmup')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lamb', default=0.1, type=float,
                    help='lambda: multiplier for added loss')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--neg-loss', action='store_true', default=False,
                    help='use loss on negative pairs')
parser.add_argument('--use-resnet18', action='store_true',
                    help='Use resnet18 model')
parser.add_argument('--use-resnet34', action='store_true',
                    help='Use resnet34 model')
parser.add_argument('--use-resnet50', action='store_true',
                    help='Use resnet50 model')
parser.add_argument('--use-inception-v3', action='store_true',
                    help='Use inception v3 model')
parser.add_argument('--use-photon-net', default=None, type=str,
                    help=' Use Photon Net')
parser.add_argument('--num-instances', default=None, type=float,
                    help='number of positive samples per sample image')
#parser.add_argument('--use-contrastive-allfeats', default=None, type=str,
#                    help='Use contrastive loss with all feature oututs, not just final')
parser.add_argument('--use-student-teacher', default=None, type=str,
                    help='Use student teacher training. Use location of clean model weights')
parser.add_argument('--use-dirty-pixel', default=None, type=str,
                    help='Use dirty pixels for training. Use location of denoiser weights')
parser.add_argument('--train-denoiser', action='store_true',
                    help='Train Denoiser')
parser.add_argument('--pil-loader', action='store_true',
                    help='Uses Pillow image loader if true, used open-cv otherwose', default=False)
parser.add_argument('--cub-training', action='store_true',
                    help='training cub classification model, uses cub data augmentation')
parser.add_argument('--spad-training', action='store_true',
                    help='training cub classification model using spad data, uses cub data augmentation')
parser.add_argument('--cars-training', action='store_true',
                    help='training cars classification model, uses cars data augmentation')
parser.add_argument('--imagenet-training', action='store_true',
                    help='training imagenet classification, uses imagenet data augmentation')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU ID to use.')
parser.add_argument('--experiment', default='../experiments', type=str,
                    help='path to save log and models folder for the experiment')
parser.add_argument('--label-file', default='train.txt', type=str,
                    help='name of the file with labels')
parser.add_argument('--mean', nargs='+', default=None)
parser.add_argument('--std', nargs='+', default=None)


networks.CLASSES = 100

writer = None
def main(args):
  # parse args
  best_acc1 = 0.0
  # tensorboard writer
  global writer
  writer = SummaryWriter(args.experiment + "/logs")

  if args.gpu >= 0:
    print("Use GPU: {}".format(args.gpu))
  else:
    print('Using CPU for computing!')

  fixed_random_seed = 2019
  torch.manual_seed(fixed_random_seed)
  np.random.seed(fixed_random_seed)
  random.seed(fixed_random_seed)


  # set up transforms for data augmentation
  mn = [float(x) for x in args.mean] if(args.mean) else [0.485, 0.456, 0.406] 
  st = [float(x) for x in args.std] if(args.std) else [0.229, 0.224, 0.225] 

  normalize = transforms.Normalize(mean=mn, std=st)
  train_transforms = get_train_transforms(normalize)
  val_transforms = get_val_transforms(normalize)
  if(args.train_denoiser):
    normalize = transforms.Normalize(mean=mn, std=st)
    train_transforms = get_denoiser_train_transforms(normalize)
    val_transforms = get_denoiser_val_transforms(normalize)
  elif(args.cub_training):
    networks.CLASSES=200
    normalize = transforms.Normalize(mean=mn, std=st)
    train_transforms = get_cub_train_transforms(normalize)
    val_transforms = get_cub_val_transforms(normalize)
  if(args.spad_training):
    networks.CLASSES=122
    normalize = transforms.Normalize(mean=mn, std=st)
    train_transforms = get_spad_train_transforms(normalize)
    val_transforms = get_spad_val_transforms(normalize)
  elif(args.cars_training):
    networks.CLASSES=196
    normalize = transforms.Normalize(mean=mn, std=st)
    train_transforms = get_cub_train_transforms(normalize)
    val_transforms = get_cub_val_transforms(normalize)
  elif(args.imagenet_training):
    networks.CLASSES=1000
    normalize = transforms.Normalize(mean=mn, std=st)
    train_transforms = get_imagenet_train_transforms(normalize)
    val_transforms = get_imagenet_val_transforms(normalize)
  if (not args.evaluate):
    print("Training time data augmentations:")
    print(train_transforms)


  model_clean=None
  model_teacher=None
  if args.use_resnet18:
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, networks.CLASSES)
  elif args.use_resnet34:
    model = torchvision.models.resnet34(pretrained=False)
    model.fc = nn.Linear(512, networks.CLASSES)
  elif args.use_resnet50:
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, networks.CLASSES)
  elif args.use_inception_v3:
    model = torchvision.models.inception_v3(pretrained=False, aux_logits=False)
    model.fc = nn.Linear(2048, networks.CLASSES)
  elif args.use_photon_net:
    model = networks.ResNetContrast(BasicBlock, [2, 2, 2, 2], networks.CLASSES)
    if(args.use_photon_net!="random"):
        model.load_state_dict(torch.load(args.use_photon_net)['state_dict'])
#  elif args.use_contrastive_allfeats:
#    model = networks.ResNetContrast2(BasicBlock, [2, 2, 2, 2], networks.CLASSES)
#    if(args.use_contrastive_allfeats!="random"):
#        model.load_state_dict(torch.load(args.use_contrastive_allfeats)['state_dict'])
  elif args.train_denoiser:
    model = UNet(3,3)
  elif args.use_dirty_pixel:
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, networks.CLASSES)
    model_clean = UNet(3,3)
    if(args.evaluate==False):
        model_clean.load_state_dict(torch.load(args.use_dirty_pixel)['state_dict'])
        model_clean = model_clean.cuda(args.gpu)
  elif args.use_student_teacher:
    model = networks.ResNetPerceptual(BasicBlock, [2, 2, 2, 2], networks.CLASSES)
    model_teacher = networks.ResNetPerceptual(BasicBlock, [2, 2, 2, 2], networks.CLASSES, teacher_model=True)
    model_teacher.load_state_dict(torch.load(args.use_student_teacher)['state_dict'])
    model_teacher = model_teacher.cuda(args.gpu)
    model_teacher.eval()
    for param in model_teacher.parameters():
      param.requires_grad = False
  else:
    print("select correct model")
    exit(0)

  criterion1 = nn.CrossEntropyLoss()
  if(args.use_student_teacher or args.train_denoiser or args.use_dirty_pixel):
    criterion2 = nn.MSELoss()
  else:
    ps = AllPositivePairSelector(balance=False)
    criterion2 = OnlineContrastiveLoss(1., ps)
  # put everthing to gpu
  if args.gpu >= 0:
    model = model.cuda(args.gpu)
    criterion1 = criterion1.cuda(args.gpu)
    #criterion3 = criterion3.cuda(args.gpu)
    criterion2 = criterion2.cuda(args.gpu)
  criterion = [criterion1, criterion2]
  #criterion = [criterion1]
  # setup the optimizer
  opt_params = model.parameters()
  if(args.use_dirty_pixel):
    opt_params = list(model.parameters()) + list(model_clean.parameters())
  optimizer = torch.optim.SGD(opt_params, args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)

  # resume from a checkpoint?
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      if(args.gpu>=0):
        checkpoint = torch.load(args.resume)
      else:
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
      #best_acc1 = checkpoint['best_acc1']


      #new_state_dict = OrderedDict()
      #model_dict = model.state_dict()
      #for k, v in checkpoint['state_dict'].items():
      #  name = k[7:] # remove `module.`
      #  if(name.startswith('fc')):
      #      continue
      #  new_state_dict[name] = v
      #model_dict.update(new_state_dict)
      #model.load_state_dict(model_dict)
      model.load_state_dict(checkpoint['state_dict'])
      if args.gpu < 0:
        model = model.cpu()
      else:
        model = model.cuda(args.gpu)
      if(args.use_dirty_pixel):
        model_clean.load_state_dict(checkpoint['model_clean_state_dict'])
        model_clean = model_clean.cuda(args.gpu)
#      # only load the optimizer if necessary
#      if (not args.evaluate):
#        args.start_epoch = checkpoint['epoch']
#        optimizer.load_state_dict(checkpoint['optimizer'])
      print("=> loaded checkpoint '{}' (epoch {}, acc1 {})"
          .format(args.resume, checkpoint['epoch'], best_acc1))
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))


  # setup dataset and dataloader
  val_dataset = IMMetricLoader(args.data_folder,
                  split='val', transforms=val_transforms, image_id=False, pil_loader=args.pil_loader, clean_image=args.train_denoiser)

  print('Validation Set Size: ', len(val_dataset))

  val_batch_size = 1 if args.train_denoiser else 50
  val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=val_batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)
  val_dataset.reset_seed()
  # evaluation
  if args.evaluate:
    print("Testing the model ...")
    cudnn.deterministic = True
    validate(val_loader, model, -1, args, model_clean)
    return
  load_clean_image = args.use_student_teacher or args.train_denoiser or args.use_dirty_pixel
  train_dataset = IMMetricLoader(args.data_folder,
                  split='train', transforms=train_transforms, label_file=args.label_file, pil_loader=args.pil_loader, clean_image=load_clean_image)
  print('Training Set Size: ', len(train_dataset))
  if(args.num_instances):
    train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    num_workers=args.workers, pin_memory=True, sampler=RandomIdentitySampler(train_dataset, num_instances=args.num_instances), drop_last=True)
  else:
    train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.batch_size, shuffle=True,
      num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

  # enable cudnn benchmark
  cudnn.enabled = True
  cudnn.benchmark = True


  if(args.train_denoiser):
    
    print("Training denoiser ...")
    for epoch in range(args.start_epoch, args.epochs):
      train_dataset.reset_seed()
      train_denoiser(train_loader, val_loader, model, criterion, optimizer, epoch, args)
    return

  model.eval()
  top1 = AverageMeter()
  top5 = AverageMeter()
  val_acc1 = validate(val_loader, model, 0, args, model_clean)
  writer.add_scalars('data/top1_accuracy',
     {"train" : top1.avg}, 0)
  writer.add_scalars('data/top5_accuracy',
     {"train" : top5.avg}, 0)
  model.train()

  # warmup the training
  if (args.start_epoch == 0) and (args.warmup_epochs > 0):
    print("Warmup the training ...")
    for epoch in range(0, args.warmup_epochs):
      acc1 = train(train_loader, val_loader, model, criterion, optimizer, epoch, "warmup", best_acc1, args, model_clean, model_teacher)

  # start the training
  print("Training the model ...")
  for epoch in range(args.start_epoch, args.epochs):
    train_dataset.reset_seed()
    # train for one epoch
    acc1 = train(train_loader, val_loader, model, criterion, optimizer, epoch, "train", best_acc1, args, model_clean, model_teacher)


    # save checkpoint
    best_acc1 = max(acc1, best_acc1)


def save_model(state, file_folder, fname='model_best.pth.tar'):
  """save model"""
  if not os.path.exists(file_folder):
    os.mkdir(file_folder)
  # skip the optimization state
  torch.save(state, os.path.join(file_folder, fname))


def train(train_loader, val_loader, model, criterion, optimizer, epoch, stage, acc, args, model_clean=None, model_teacher=None):
  """Training the model"""
  assert stage in ["train", "warmup"]
  # adjust the learning rate
  num_iters = len(train_loader)
  lr = 0.0

  # set up meters
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  best_acc1 = acc
  # switch to train mode
  model.train()
  if(args.use_dirty_pixel):
    model_clean.train()
  end = time.time()
  iter_eval = len(train_loader)//args.eval_count
  for i, (images, target, image_id) in enumerate(train_loader):
    #input = input2.view(input2.shape[0]*input2.shape[1], input2.shape[2], input2.shape[3], input2.shape[4])
    #target = target2.repeat(5,1).T.reshape(-1)
    #image_id = image_id2.repeat(5,1).T.reshape(-1)
    # adjust the learning rate
    if stage == "warmup":
      # warmup: linear scaling
      lr = (epoch * num_iters + i) / float(
        args.warmup_epochs * num_iters) * args.lr
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = 0.0
    else:
      # cosine learning rate decay
      lr = 0.5 * args.lr * (1 + math.cos(
        (epoch * num_iters + i) / float(args.epochs * num_iters) * math.pi))
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = args.weight_decay

    # measure data loading time
    data_time.update(time.time() - end)
    if args.gpu >= 0:
      images = images.cuda(args.gpu, non_blocking=True)
      target = target.cuda(args.gpu, non_blocking=True)
      image_id = image_id.cuda(args.gpu, non_blocking=True)
    if(args.use_student_teacher or args.use_dirty_pixel):
        input = images[:,0,:,:,:]
        input_clean = images[:,1,:,:,:]
    else:
        input = images
    lamb = args.lamb
    if(args.use_resnet18 or args.use_resnet34 or args.use_resnet50 or args.use_inception_v3):
        output = model(input)
        loss1 = criterion[0](output, target)
        loss = loss1
    elif(args.use_dirty_pixel):
        output_clean = model_clean(input)
        output = model(output_clean)
        loss1 = criterion[0](output, target)
        loss = loss1
        if(lamb>0):
            loss2 = criterion[1](output_clean, input_clean)
            loss = loss1 + loss2*lamb
    elif(args.use_student_teacher):
        output, feat = model(input)
        loss1 = criterion[0](output, target)
        loss = loss1
        if(lamb>0):
            output_clean, feat_clean = model_teacher(input_clean)
            loss2 = 0
            for x in range(len(feat)):
                loss2 += criterion[1](feat[x], feat_clean[x]) 
        loss = loss1 + loss2*lamb
    else:
        output, feat = model(input)
        loss1 = criterion[0](output, target)
        loss = loss1
        if(lamb>0):
            if(args.use_photon_net):    
                loss2 = criterion[1](feat, image_id, neg_loss=args.neg_loss)
#            elif(args.use_contrastive_allfeats):    
#                loss2 = 0
#                for x in range(len(feat)):
#                    loss2 += criterion[1](feat[x], image_id, neg_loss=args.neg_loss) 
            else:
                print('Lambda is not used')
                exit(0)
            loss = loss1 + loss2*lamb


    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    losses.update(loss.item(), input.size(0))
    top1.update(acc1[0], input.size(0))
    top5.update(acc5[0], input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    if args.gpu >= 0:
        torch.cuda.synchronize()
    batch_time.update(time.time() - end)
    end = time.time()

    # printing
    if i % args.print_freq == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
        'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
        'Acc@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
         epoch + 1, i, len(train_loader), batch_time=batch_time,
         data_time=data_time, loss=losses, top1=top1, top5=top5))
      # log loss / lr
      if stage == "train":
        writer.add_scalar('data/training_loss',
          losses.val, epoch * num_iters + i)
        writer.add_scalar('data/learning_rate',
          lr, epoch * num_iters + i)

    if (i+1)%iter_eval==0:
        # evaluate on validation set
        val_acc1 = validate(val_loader, model, epoch*args.eval_count+(i+1)/iter_eval, args, model_clean)
        # log top-1/5 acc
        writer.add_scalars('data/top1_accuracy',
          {"train" : top1.avg}, epoch*args.eval_count+(i+1)/iter_eval)
        writer.add_scalars('data/top5_accuracy',
          {"train" : top5.avg}, epoch*args.eval_count+(i+1)/iter_eval)
        top1 = AverageMeter()
        top5 = AverageMeter()
        model.train()
        if(args.use_dirty_pixel):
            model_clean.train()
        if(best_acc1<val_acc1):
            best_acc1 = val_acc1
            state_dic = {
              'epoch': -1,
              'best_acc1': best_acc1,
            }
            state_dic['state_dict']=model.state_dict()
            if(args.use_dirty_pixel):
                state_dic['model_clean_state_dict']=model_clean.state_dict()
            save_model(state_dic, args.experiment + "/models")

  # print the learning rate
  print("[Stage {:s}]: Epoch {:d} finished with lr={:f}".format(
            stage, epoch + 1, lr))
  return best_acc1



def validate(val_loader, model, val_iter, args, model_clean=None):
  """Test the model on the validation set"""
  batch_time = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode (autograd will still track the graph!)
  model.eval()
  if(model_clean):
    model_clean.eval()
  # disable/enable gradients
  grad_flag = False
  with torch.set_grad_enabled(grad_flag):
    end = time.time()
    # loop over validation set
    for i, (input, target) in enumerate(val_loader):
      if args.gpu >= 0:
        input = input.cuda(args.gpu, non_blocking=False)
        target = target.cuda(args.gpu, non_blocking=False)

      if(args.use_dirty_pixel):
        output_clean = model_clean(input)
        output = model(output_clean)
      else:
        output = model(input)
      # test time augmentation (minor performance boost)
      if args.evaluate:
        flipped_input = torch.flip(input, (3,))
        if(args.use_dirty_pixel):
          output_clean = model_clean(flipped_input)
          flipped_output = model(output_clean)
        else:
          flipped_output = model(flipped_input)
        #flipped_output = model(flipped_input)
        output = 0.5 * (output + flipped_output)

      # measure accuracy and record loss
      acc1, acc5 = accuracy(output, target, topk=(1, 5))
      top1.update(acc1[0], input.size(0))
      top5.update(acc5[0], input.size(0))

      # measure elapsed time
      if args.gpu >= 0:
        torch.cuda.synchronize()
      batch_time.update(time.time() - end)
      end = time.time()

      # printing
      if i % args.print_freq == 0:
        print('Test: [{0}/{1}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
          'Acc@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
           i, len(val_loader), batch_time=batch_time,
           top1=top1, top5=top5))

  print('******Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
            top1=top1, top5=top5))

  if (not args.evaluate):
    # log top-1/5 acc
    writer.add_scalars('data/top1_accuracy',
      {"val" : top1.avg}, val_iter)
    writer.add_scalars('data/top5_accuracy',
      {"val" : top5.avg}, val_iter)

  return top1.avg

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
  args = parser.parse_args()
  print(args)
  main(args)
