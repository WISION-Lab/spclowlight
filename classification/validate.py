import torch
import time
from utils import AverageMeter, accuracy

def validate_model(val_loader, model, val_iter, args, writer, model_clean=None):
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
