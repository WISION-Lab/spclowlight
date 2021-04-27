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
from utils import AverageMeter, accuracy
from pairs_gen import AllPositivePairSelector
from collections import defaultdict
from losses import OnlineContrastiveLoss
from torchvision.models.resnet import BasicBlock, Bottleneck
import networks
from metric_sampler import RandomIdentitySampler
from unet import UNet
from denoiser import *
from train import train_model
from validate import validate_model

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
parser.add_argument('--use-resnet18', default=None, type=str,
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
parser.add_argument('--val-label-file', default='val.txt', type=str,
                    help='name of the validation file with labels')
parser.add_argument('--mean', nargs='+', default=None)
parser.add_argument('--std', nargs='+', default=None)


networks.CLASSES = 100

def main(args):
  # parse args
  best_acc1 = 0.0
  # tensorboard writer
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
    if(args.use_resnet18!="random"):
        model.load_state_dict(torch.load(args.use_resnet18)['state_dict'])
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
                  split='val', transforms=val_transforms, image_id=False, pil_loader=args.pil_loader, clean_image=args.train_denoiser, label_file=args.val_label_file)

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
    validate_model(val_loader, model, -1, args, writer, model_clean)
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
      train_denoiser(train_loader, val_loader, model, criterion, optimizer, epoch, args, writer)
    return

  model.eval()
  top1 = AverageMeter()
  top5 = AverageMeter()
  val_acc1 = validate_model(val_loader, model, 0, args, writer, model_clean)
  writer.add_scalars('data/top1_accuracy',
     {"train" : top1.avg}, 0)
  writer.add_scalars('data/top5_accuracy',
     {"train" : top5.avg}, 0)
  model.train()

  # warmup the training
  if (args.start_epoch == 0) and (args.warmup_epochs > 0):
    print("Warmup the training ...")
    for epoch in range(0, args.warmup_epochs):
      acc1 = train_model(train_loader, val_loader, model, criterion, optimizer, epoch, "warmup", best_acc1, args, writer, model_clean, model_teacher)

  # start the training
  print("Training the model ...")
  for epoch in range(args.start_epoch, args.epochs):
    train_dataset.reset_seed()
    # train for one epoch
    acc1 = train_model(train_loader, val_loader, model, criterion, optimizer, epoch, "train", best_acc1, args, writer, model_clean, model_teacher)


    # save checkpoint
    best_acc1 = max(acc1, best_acc1)



if __name__ == '__main__':
  args = parser.parse_args()
  print(args)
  main(args)
