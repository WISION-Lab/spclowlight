import numpy as np
import os, sys
import cv2

# Original Clean Images
BASE = '/srv/home/bhavya/datasets/cub/CUB_200_2011/'
# Generated Images for SPC
BASENEW = BASE.replace('CUB_200_2011', 'CUB_200_2011_average60')
# number of frames to average
frames = [1, 2, 5, 10, 30, 60]

# Settings for MiniPaces and COCO dataset
#BASE = '/srv/home/bhavya/datasets/miniplaces2/data/'
#BASE = '/srv/home/bhavya/datasets/COCO/coco17/train2017/'
#BASENEW = BASE.replace('miniplaces2/data', 'miniplaces2/data_average60')
#BASENEW = BASE.replace('coco17', 'coco17_binary5')
if not os.path.exists(BASENEW):
    os.makedirs(BASENEW)

imtxt = BASE + 'train.txt'
im_list = open(imtxt, 'r').readlines()
#im_list = os.listdir(BASE)

# Calculate Mean and Std Deviation of generated images
mn=np.zeros(3)
mn2=np.zeros(3)
TOTAL_FRAMES=60
start=0
end=len(im_list)
if(len(sys.argv)>1):
    start = int(sys.argv[1])
    end = int(sys.argv[2])

for line in im_list[start:end]:
    l = line.strip().split(' ')
    im_name = 'images/'+l[0]
    print(im_name)
    
    # Reading original clean image
    im = cv2.imread(BASE+im_name)
    count_shape = im.shape + (-1,)
    # Simulate low light conditions by scaling pixel values by 1/1000.
    # Poisson process to simulate photon arrival
    photon_counts = np.random.poisson(im.flatten()/1000.0, size=(TOTAL_FRAMES, im.size)).T
    photon_counts = photon_counts.reshape(count_shape)
    # Photon counts is converted to binary frames
    b_counts=np.where(photon_counts>=1, 1, 0)
    
    for i in range(1):
        for fil in frames:
            recon_image = np.mean(b_counts[:,:,:,fil*i:fil*(i+1)], axis=3)
            mn += np.mean(recon_image, axis=(0,1))
            mn2 += np.mean(recon_image**2, axis=(0,1))
            #outfile = BASENEW.replace('60', str(fil)) + str(i) + '/' + im_name.replace('jpg', 'png')
            outfile = BASENEW.replace('60', str(fil)) + 'images/' + str(i) + '/' + l[0].replace('jpg', 'png')
            directory = os.path.dirname(outfile)
            if not os.path.exists(directory):
                os.makedirs(directory)
            cv2.imwrite(outfile, recon_image*255.0)


size = end-start
mn = mn/size
mn2 = mn2/size
st = np.sqrt( mn2 - (mn**2) )
print(st)

