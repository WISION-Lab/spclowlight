import torch
import time
import math
import os
from utils import AverageMeter, accuracy
from validate import validate_model

def train_model(train_loader, val_loader, model, criterion, optimizer, epoch, stage, acc, args, writer, model_clean=None, model_teacher=None):
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
        val_acc1 = validate_model(val_loader, model, epoch*args.eval_count+(i+1)/iter_eval, args, writer, model_clean)
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



def save_model(state, file_folder, fname='model_best.pth.tar'):
  """save model"""
  if not os.path.exists(file_folder):
    os.mkdir(file_folder)
  # skip the optimization state
  torch.save(state, os.path.join(file_folder, fname))



