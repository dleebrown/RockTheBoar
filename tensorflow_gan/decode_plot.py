import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from skimage.io import imread

#---------------------------------------------------------------------
#Decoding

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order= 'F')

#-----------------------------------------------------------------------

def plotMask(img, mask, imtitle):

	fig, ax = plt.subplots(1,3, figsize = (15, 5))
	ax = ax.ravel()

	ax[0].imshow(img)
	ax[1].imshow(mask > 0.5, cmap = 'gray')

	ax[2].imshow(img)
	ax[2].imshow(mask>0.5, cmap = 'gray', alpha = 0.3)
	fig.suptitle(imtitle)
	plt.savefig(outputDir+'predict_'+imtitle)    
	plt.show() 


#-----------------------------------------------------------------------

#saved_csv = '../../test_output_noF.csv'
saved_csv = '../../ayyF_encode.csv'
imageDir = '../../input/train/'
outputDir = '../../plots/'

#masks = pd.read_csv(saved_csv)
#print(masks)

#num_masks = masks.shape[0]
#print('Total masks to encode/decode =', num_masks)


#for i in masks.itertuples():
chunksize = 1

for chunk in pd.read_csv(saved_csv, chunksize=chunksize):

    for i in chunk.itertuples():



    #for i in masks.itertuples(1):

        mask = rle_decode(i.rle_mask,(1280,1918))

        #print(i.img)
        #img = imread(imageDir+i.img)
        #plotMask(img,mask,i.img)

        plt.imshow(mask)

        plt.show()
        raw_input()
