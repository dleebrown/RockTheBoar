import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from skimage.io import imread




#-----------------------------------------------------------------------
#Encoding

def rle_encode_C(mask_image):
    # Default C ordering
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs



def rle_encode_F(mask_image):
    pixels = mask_image.flatten(order = 'F')
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

#-----------------------------------------------------------------------


#---------------------------------------------------------------------
#Decoding - default C

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
	ax[1].imshow(mask > 0.5, cmap = 'gray')

	ax[2].imshow(img)
	ax[2].imshow(mask>0.5, cmap = 'gray', alpha = 0.3)
	fig.suptitle(imtitle)
	plt.savefig(outputDir+'predict_'+imtitle)    
	plt.show() 


#-----------------------------------------------------------------------

saved_csv = '../../test_out_trim.csv'
#saved_csv = '../../test_output_noF.csv'
output_csv = '../../ayyF_encode1.csv'
#masks = pd.read_csv(saved_csv)
#print(masks)

#num_masks = masks.shape[0]
#print('Total masks to encode/decode =', num_masks)

output_file = open(output_csv, mode='w')
output_file.write('img,rle_mask\n')

#for i in masks.itertuples():
chunksize = 1000

for chunk in pd.read_csv(saved_csv, chunksize=chunksize):

    for i in chunk.itertuples():
       #print i
       mask = rle_decode(i.rle_mask,(1280,1918))

       rle = rle_encode_F(mask)
       rle_string = rle_to_string(rle)
       #print i
       #print(i.img +','+ rle_string)
       #print(50*'-')
       output_file.write(i.img+','+rle_string+'\n')


output_file.close()
#plt.show()
