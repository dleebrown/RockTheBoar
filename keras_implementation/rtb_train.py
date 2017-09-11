# coding: utf-8
import numpy as np
import keras
import keras.backend as K

import io_functions
import simple_model

import pandas as pd
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------
# Hyper-parameters
num_train = 10  # while protoyping, otherwise: len(ids_train)
learning_rate = 1e-3
num_epochs = 2
batch_size = 2
#-----------------------------------------------------------------------
# load model architecture
model = simple_model.model()
#-----------------------------------------------------------------------
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

smooth = 1.

# From here: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

model.compile(Adam(lr=learning_rate), bce_dice_loss, metrics=['accuracy', dice_coef])

# load data
X_train, y_train, X_val, y_val = io_functions.load_data(input_folder = '../../input/', num_train = num_train)

#Training
history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), batch_size=batch_size)

# Save the trained model and training history in hdf5 and npy respectively
io_functions.saveModel(TrainedModel = model, TrainingHistory = history, fileOut = 'trainedModel')







#-----------------------------------------------------------------------
print 50*'-'
# Testing

idx = 0
x = X_val[idx]

fig, ax = plt.subplots(5,3, figsize=(16, 16))
ax = ax.ravel()

cmaps = ['Reds', 'Greens', 'Blues']
for i in range(x.shape[-1]):
    ax[i].imshow(x[...,i], cmap='gray') #cmaps[i%3])
    ax[i].set_title('channel {}'.format(i))


X_mean = np.load('X_mean.npy')
X_std = np.load('X_std.npy')


ax[-3].imshow((x[...,:3] * X_std[0,...,:3] + X_mean[0,...,:3]) / 255.)
ax[-3].set_title('X')

ax[-2].imshow(y_train[idx,...,0], cmap='gray')
ax[-2].set_title('y')

y_pred = model.predict(x[None]).squeeze()
ax[-1].imshow(y_pred, cmap='gray')
ax[-1].set_title('y_pred')


plt.imshow(y_pred > 0.5, cmap='gray')
plt.savefig('plots/out.png')
#plt.show()


# plot loss, acc, dice coeff
pd.DataFrame(history.history)[['dice_coef', 'val_dice_coef']].plot()
plt.savefig('plots/dice.png')

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.savefig('plots/loss.png')

pd.DataFrame(history.history)[['acc', 'val_acc']].plot()
plt.savefig('plots/acc.png')


print y_pred.shape
print y_train.shape
print x[None].shape

#-----------------------------------------------------------------------
plt.clf()

def plotMask(img, mask):

	fig, ax = plt.subplots(1,3, figsize = (15, 5))
	ax = ax.ravel()

	ax[0].imshow((img[...,:3] * X_std[0,...,:3] + X_mean[0,...,:3]) / 255.)
	
	#ax[0].imshow(img, cmap = 'gray')
	
	ax[1].imshow(mask > 0.5, cmap = 'gray')

	ax[2].imshow((img[...,:3] * X_std[0,...,:3] + X_mean[0,...,:3]) / 255., cmap = 'jet')
	ax[2].imshow(mask>0.5, cmap = 'gray', alpha = 0.5)
	
	plt.savefig('plots/testPredict.png')    
	#plt.show() 

#-----------------------------------------------------------------------

plotMask(x, y_pred)



#-----------------------------------------------------------------------
#Encoding

def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

#-----------------------------------------------------------------------


rle = rle_encode(y_pred)

rle_string = rle_to_string(rle)

print rle_string



