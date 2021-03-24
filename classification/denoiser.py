import math
from utils import AverageMeter


def train_denoiser(train_loader, val_loader, model, criterion, optimizer, epoch, args):
  """Training the model"""
  # adjust the learning rate
  num_iters = len(train_loader)
  lr = 0.0

  # set up meters
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  # switch to train mode
  model.train()

  end = time.time()
  for i, (images, target, image_id) in enumerate(train_loader):
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
    input = images[:,0,:,:,:]
    input_clean = images[:,1,:,:,:]
    output = model(input)
    loss = criterion[1](output, input_clean)

    losses.update(loss.item(), input.size(0))
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
        'Loss {loss.val:.2f} ({loss.avg:.2f}))'.format(
         epoch + 1, i, len(train_loader), batch_time=batch_time,
         data_time=data_time, loss=losses))
      writer.add_scalar('data/training_loss',
          losses.val, epoch * num_iters + i)
      writer.add_scalar('data/learning_rate',
          lr, epoch * num_iters + i)


  avg_psnr = validate_denoiser(val_loader, model, criterion, args)
  print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
  model.train()
  state_dic = {}
  state_dic['state_dict']=model.state_dict()
  save_model(state_dic, args.experiment + "/models")

  print("Epoch {:d} finished with lr={:f}".format(epoch + 1, lr))
  return



def validate_denoiser(val_loader, model, criterion, args):
    print("Validating Denoiser")
    model.eval()
    avg_psnr = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args.gpu >= 0:
                images = images.cuda(args.gpu, non_blocking=False)
                target = target.cuda(args.gpu, non_blocking=False)
            input = images[:,0,:,:,:]
            input_clean = images[:,1,:,:,:]

            prediction = model(input)
            mse = criterion[1](prediction, input_clean)
            psnr = 10 * math.log10(1 / mse.item())
            avg_psnr += psnr
    return avg_psnr/len(val_loader)


