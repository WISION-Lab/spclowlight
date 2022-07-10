import numpy as np
import os, sys
import cv2
import torch
# Original Clean Images
BASE = '/srv/home/bhavya/datasets/reds/'
# Generated Images for SPC
BASENEW = BASE.replace('reds', 'reds_dark')
# number of frames to average
#frames = [1, 4, 16, 64, 256]
frames = [1]

# Settings for MiniPaces and COCO dataset
#BASE = '/srv/home/bhavya/datasets/miniplaces2/data/'
#BASE = '/srv/home/bhavya/datasets/COCO/coco17/train2017/'
#BASENEW = BASE.replace('miniplaces2/data', 'miniplaces2/data_average60')
#BASENEW = BASE.replace('coco17', 'coco17_binary5')

imtxt = BASE + 'train.txt'
im_list = open(imtxt, 'r').readlines()

#TOTAL_FRAMES=256
start=0
end=len(im_list)
if(len(sys.argv)>1):
    start = int(sys.argv[1])
    end = int(sys.argv[2])


crf_path = 'crf.pt'
crf_inv = torch.from_numpy(torch.load(crf_path)).squeeze_()  # 256 x 3

# 1st order approximation at RGB=250 to regularize extreme responses at RGB>250
diff = (crf_inv[251] - crf_inv[249])/2
for i in range(251, 256):
    crf_inv[i] = crf_inv[i-1] + diff

# frame interpolation, etc.

#buffer_tensor = buffer_tensor.permute(0,2,3,1).reshape(-1, C).mul_(255).add_(0.5).clamp_(0, 255).long() # bad naming to save GPU memory
#buffer_tensor = torch.gather(crf_inv, 0, buffer_tensor).reshape(-1,H,W,C).permute(0,3,1,2)

filenames = []
for i in range(100, 109):
    filenames.append('00000'+str(i)+'.png')
    #filenames.append('00000100.png')

#filenames = ["00000100.png", "00000101.png", "00000102.png", "00000103.png", "00000104.png", "00000105.png", "00000106.png", "00000107.png", "00000108.png"]
#filenames = ["00000100.png"]*100#, "00000100.png", "00000100.png", "00000100.png", "00000100.png", "00000100.png", "00000100.png", "00000100.png", "00000100.png"]
for line in im_list[start:end]:
    l = line.strip().split()
    #im_name = 'images/'+l[0]
    #im_name = l[0].replace('data/','')
    im_name = l[0]
    file_name = im_name.split('/')[-1]
    if(file_name!='00000100.png'):
         continue
    all_photon_counts = []
    for filename in filenames:
        im_filename = '/'.join(im_name.split('/')[:-1])+'/'+filename
        print(im_filename)
        # Reading original clean image
        im = cv2.imread(BASE+im_filename)

        im = im.transpose(2,0,1)
        crf_im = np.zeros_like(im).astype(np.float64)
        crf_im[0,:,:] = np.take(crf_inv[:,0], im[0,:,:])
        crf_im[1,:,:] = np.take(crf_inv[:,1], im[1,:,:])
        crf_im[2,:,:] = np.take(crf_inv[:,2], im[2,:,:])
        crf_im = crf_im.transpose(1,2,0)
        im = crf_im

        #count_shape = im.shape + (-1,)
        # Simulate low light conditions by scaling pixel values by 1/1000.
        # Poisson process to simulate photon arrival
        exp_time = 1/120.
        # sim = 200, med = 100, dark = 50
        photon_counts = np.random.poisson(im * 50 * exp_time)
        all_photon_counts.append(photon_counts)
    all_photon_counts = np.array(all_photon_counts)
    #for idx in [0, 1, 40, 200]:#range(200,210):
    for idx, filename in enumerate(filenames[4:]):
            if(idx!=0):
                continue
            recon_image = np.sum(all_photon_counts[4-idx:4+(idx)+1,:,:,:]*1.0, axis=0)
            r_i = np.sort(recon_image.flatten())
            scale = r_i[int(r_i.shape[0]*0.97)]
            print(scale)
            #photon_counts = photon_counts.reshape(count_shape)
            # Photon counts is converted to binary frames
            #b_counts=np.where(photon_counts>=1, 1, 0)
            #recon_image = np.mean(b_counts[:,:,:,0:fil], axis=3)
            recon_image = np.clip(recon_image*(1./scale), 0, 1.)
            recon_image = np.power(recon_image, 1./2.2)

            #recon_image = np.clip(recon_image, 0, 255)
            #outfile = BASENEW.replace('60', str(fil)) + str(i) + im_name.replace('JPEG', 'png')
            im_filename = '/'.join(im_name.split('/')[:-1]) + '/' + filenames[4]
            outfile = BASENEW.replace('reds_dark', 'reds_dark'+str(idx)) + '/' + im_filename
            directory = os.path.dirname(outfile)
            if not os.path.exists(directory):
                os.makedirs(directory)
            cv2.imwrite(outfile, recon_image*255)


