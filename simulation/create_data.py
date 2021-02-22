import numpy as np
import os, sys
import cv2
# Original Clean Images
BASE = '/srv/home/bhavya/datasets/imagenet/'
# Generated Images for SPC
BASENEW = BASE.replace('imagenet', 'imagenet_generated/average60')
# number of frames to average
frames = [4, 16, 64, 256]
#frames = [4, 8, 16, 32, 64, 128]
#frames = [16, 32, 64, 128, 256, 9, 84]
#frames = [17, 52, 90, 80, 160, 90, 32, 21, 46, 18, 56]
#frames = [3, 4, 6, 7, 8, 14, 20, 28, 40, 42, 85, 120, 170, 240]
#frames = [2,3,4,5]
#frames = [1]

# Settings for MiniPaces and COCO dataset
#BASE = '/srv/home/bhavya/datasets/miniplaces2/data/'
#BASE = '/srv/home/bhavya/datasets/COCO/coco17/train2017/'
#BASENEW = BASE.replace('miniplaces2/data', 'miniplaces2/data_average60')
#BASENEW = BASE.replace('coco17', 'coco17_binary5')

imtxt = BASE + 'train.txt'
im_list = open(imtxt, 'r').readlines()
#im_list = os.listdir(BASE)

# Calculate Mean and Std Deviation of generated images
#mn=np.zeros(3)
#mn2=np.zeros(3)
TOTAL_FRAMES=256
start=0
end=len(im_list)
if(len(sys.argv)>1):
    start = int(sys.argv[1])
    end = int(sys.argv[2])

for line in im_list[start:end]:
    l = line.strip().split()
    #im_name = 'images/'+l[0]
    #im_name = l[0].replace('data/','')
    im_name = l[0]
    print(im_name)
    # Reading original clean image
    im = cv2.imread(BASE+im_name)
    count_shape = im.shape + (-1,)
    
    for i in range(1):
#        outfile = BASENEW.replace('60', str(256)) + str(i) + '/' + im_name.replace('jpg', 'png')
#        if(os.path.exists(outfile)):
#            continue

        # Simulate low light conditions by scaling pixel values by 1/1000.
        # Poisson process to simulate photon arrival
        photon_counts = np.random.poisson(im.flatten()/1000.0, size=(TOTAL_FRAMES, im.size)).T
        photon_counts = photon_counts.reshape(count_shape)
        # Photon counts is converted to binary frames
        b_counts=np.where(photon_counts>=1, 1, 0)
        for fil in frames:
            recon_image = np.mean(b_counts[:,:,:,0:fil], axis=3)
            #recon_image = im/1000.
            #mn += np.mean(recon_image, axis=(0,1))
            #mn2 += np.mean(recon_image**2, axis=(0,1))
            #outfile = BASENEW + im_name.replace('jpg', 'png')
            outfile = BASENEW.replace('60', str(fil)) + str(i) + im_name.replace('JPEG', 'png')
            #outfile = BASENEW.replace('60', str(fil)) + 'images/' + str(i) + '/' + im_name[7:].replace('jpg', 'png')
            #print(outfile)
            #exit(0)
            directory = os.path.dirname(outfile)
            if not os.path.exists(directory):
                os.makedirs(directory)
            cv2.imwrite(outfile, recon_image*255.0)


#size = end-start
#mn = mn/size
#mn2 = mn2/size
#st = np.sqrt( mn2 - (mn**2) )
#print(st)

