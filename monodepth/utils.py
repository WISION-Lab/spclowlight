import matplotlib
import matplotlib.cm
import numpy as np
import torch
import torchvision.utils as vutils    
import cv2
import os
def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1))


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a01 = (thresh < 1.01   ).mean()
    a05 = (thresh < 1.05   ).mean()
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    abs_rel_median = np.median(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10, a01, a05, abs_rel_median


def scale_up(scale, images):
    from skimage.transform import resize
    scaled = []
    
    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append( resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True ) )

    return np.stack(scaled)


def evaluate_model(model, test_loader, args):
    print('Evaluating...')
    model.eval()
    predictions = []
    #maxDepth = 1000
    testSetDepths = []
    #testSetDepths = np.load('/srv/home/bhavya/datasets/NYU_others/eigen_test_depth.npy')
    crop = np.load(args.data_folder + '/../eigen_test_crop.npy')
    crop = crop//2
    OUTPUT_DIR = 'output_images_normed/'
    if(not os.path.exists(OUTPUT_DIR)):
        os.makedirs(OUTPUT_DIR)
    for i, sample_batched in enumerate(test_loader):
        #image = torch.autograd.Variable(sample_batched['image'].cuda())
        #depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
        image = sample_batched['image'].cuda()
        depth = sample_batched['depth'].cuda()
        output = model(image)
        prediction = output.data.cpu().numpy()
        #output_flip = model(image.flip(2))
        #prediction_flip = output_flip.flip(2).data.cpu().numpy()
        #predictions.append(prediction*0.5 +  prediction_flip*0.5)
        predictions.append(prediction)
        depthoutput = depth.data.cpu().numpy()
        testSetDepths.append(depthoutput)
        if(True):
            #c = colorize(vutils.make_grid(DepthNorm(output).data, nrow=6, normalize=False), vmin=None, vmax=None, cmap='jet')
            #d = colorize(vutils.make_grid(depth.data, nrow=6, normalize=False), vmin=None, vmax=None, cmap='jet')
            vmin = DepthNorm(depth).data.cpu().numpy().min()
            vmax = DepthNorm(depth).data.cpu().numpy().max()
            c = colorize(vutils.make_grid(output.data, nrow=6, normalize=False), vmin=vmin, vmax=vmax, cmap='rainbow')
            d = colorize(vutils.make_grid(DepthNorm(depth).data, nrow=6, normalize=False), vmin=vmin, vmax=vmax, cmap='rainbow')
            cv2.imwrite(OUTPUT_DIR+str(i)+'.png', c.transpose((1,2,0)))
            cv2.imwrite(OUTPUT_DIR+str(i)+'d.png', d.transpose((1,2,0)))
        del image
        del depth
        del output
    predictions = np.vstack(predictions)
    #predictions = predictions.squeeze()
    #predictions = scale_up(2, predictions)
    predictions = predictions[:,:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
    #predictions = np.clip(predictions, 0, maxDepth)*10./maxDepth
    #print(predictions.shape)
    testSetDepths = np.vstack(testSetDepths)
    testSetDepths = DepthNorm(testSetDepths)
    testSetDepths = testSetDepths[:,:,crop[0]:crop[1]+1,crop[2]:crop[3]+1]
    #print(testSetDepths.shape)
    e = compute_errors(predictions, testSetDepths)
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10} , {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'a01', 'a05', 'rel', 'rel(median)', 'rms', 'log_10'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[6], e[7], e[3], e[8], e[4],e[5]))

