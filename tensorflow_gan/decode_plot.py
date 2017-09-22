import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from skimage.io import imread
#from skimage.io import imshow




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
    return img.reshape(shape)

#-----------------------------------------------------------------------

def plotMask(img, mask, imtitle):

	fig, ax = plt.subplots(1,3, figsize = (15, 5))
	ax = ax.ravel()

	ax[0].imshow(img)
	
	#ax[0].imshow(img, cmap = 'gray')
	
	ax[1].imshow(mask > 0.5, cmap = 'gray')

	ax[2].imshow(img)
	ax[2].imshow(mask>0.5, cmap = 'gray', alpha = 0.3)
	fig.suptitle(imtitle)
	plt.savefig(outputDir+'predict_'+imtitle)    
	plt.show() 


#-----------------------------------------------------------------------

saved_csv = '../../test_output_noF.csv'
imageDir = '../../input/train/'
outputDir = '../../plots/'

masks = pd.read_csv(saved_csv)
print(masks)

num_masks = masks.shape[0]
print('Total masks to encode/decode =', num_masks)

for i in masks.itertuples():

    mask = rle_decode(i.rle_mask,(1280,1918))


    #plt.figure()
    #plt.imshow(mask > 0.5, cmap='gray', alpha = 0.5)
    #plt.title(i.img)
   
    print(i.img)
    img = imread(imageDir+i.img)
    #plt.figure()
    plotMask(img,mask,i.img)

    #plt.title(i.img)
   

plt.show()
