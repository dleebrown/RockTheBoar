import numpy as np
import pandas as pd
from os.path import join
from skimage.io import imread
from skimage.transform import downscale_local_mean
from sklearn.model_selection import train_test_split
from keras.models import load_model

#-----------------------------------------------------------------------

def load_data(input_folder, num_train):
	#input_folder = join('../..', 'input')
	print input_folder
	df_mask = pd.read_csv(join(input_folder, 'train_masks.csv'), usecols=['img'])
	ids_train = df_mask['img'].map(lambda s: s.split('_')[0]).unique()

	imgs_idx = list(range(1, 17))

	load_img = lambda im, idx: imread(join(input_folder, 'train', '{}_{:02d}.jpg'.format(im, idx)))
	load_mask = lambda im, idx: imread(join(input_folder, 'train_masks', '{}_{:02d}_mask.gif'.format(im, idx)))
	resize = lambda im: downscale_local_mean(im, (4,4) if im.ndim==2 else (4,4,1))
	mask_image = lambda im, mask: (im * np.expand_dims(mask, 2))

#-----------------------------------------------------------------------

	# Load data for position id=1
	X = np.empty((num_train, 320, 480, 12), dtype=np.float32)
	y = np.empty((num_train, 320, 480, 1), dtype=np.float32)

	idx = 1 # Rotation index
	for i, img_id in enumerate(ids_train[:num_train]):
	    imgs_id = [resize(load_img(img_id, j)) for j in imgs_idx]
	    # Input is image + mean image per channel + std image per channel
	    X[i, ..., :9] = np.concatenate([imgs_id[idx-1], np.mean(imgs_id, axis=0), np.std(imgs_id, axis=0)], axis=2)
	    y[i] = resize(np.expand_dims(load_mask(img_id, idx), 2)) / 255.
	    del imgs_id # Free memory


	print('X.shape', 'y.shape', X.shape, y.shape)

#-----------------------------------------------------------------------

	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#-----------------------------------------------------------------------

	# Concat overall y info to X
	# This is important as the kernels of CNN used below has no information of its location
	y_train_mean = y_train.mean(axis=0)
	y_train_std = y_train.std(axis=0)
	y_train_min = y_train.min(axis=0)

	y_features = np.concatenate([y_train_mean, y_train_std, y_train_min], axis=2)

	X_train[:, ..., -3:] = y_features
	X_val[:, ..., -3:] = y_features

	# Normalize input and output
	X_mean = X_train.mean(axis=(0,1,2), keepdims=True)
	X_std = X_train.std(axis=(0,1,2), keepdims=True)

	X_train -= X_mean
	X_train /= X_std

	X_val -= X_mean
	X_val /= X_std
	
	print(X_mean)
	print(X_std)

	np.save('X_mean.npy', X_mean)
	np.save('X_std.npy', X_std)

	return X_train, y_train, X_val, y_val
#-----------------------------------------------------------------------

def saveModel(TrainedModel, TrainingHistory, fileOut):

    train_loss = TrainingHistory.history['loss']
    val_loss = TrainingHistory.history['val_loss']
    train_acc = TrainingHistory.history['acc']
    val_acc = TrainingHistory.history['val_acc']

    epochs = np.arange(1, np.size(train_loss)+1)

    training_hist = np.vstack([epochs, train_loss, val_loss, train_acc, val_acc])

    # fileOut =

    TrainedModel.save(fileOut+'.hdf5')
    np.save(fileOut+'.npy', training_hist)

    print('final acc - train and val')
    print(train_acc[-1], val_acc[-1])



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



def exportSubmissionFile(fileOut):
    #print("Generating submission file")
    #df = pd.DataFrame({'img': names, 'rle_mask': rles})
    #df.to_csv(fileOut+'.csv.gz', index=False, compression='gzip')
   

    file_name = get_car_image_files(image_id)[0].split("/")[-1]
    mask_rle = train_masks_df[train_masks_df['img'] == file_name]["rle_mask"].iloc[0]
    assert rle_to_string(rle_encode(mask)) == mask_rle, "Mask rle don't match"
    
    print("Mask rle match!")


def get_car_image_files(car_image_id, get_mask=False):
    if get_mask:
        if car_image_id in masks_ids:
            return [train_masks_data + "/" + s for s in train_masks_files if car_image_id in s]
        else:
            raise Exception("No mask with this ID found")
    elif car_image_id in train_ids:
        return [train_data + "/" + s for s in train_files if car_image_id in s]
    elif car_image_id in test_ids:
        return [test_data + "/" + s for s in test_files if car_image_id in s]
    raise Exception("No image with this ID found")
    
def get_image_matrix(image_path):
    img = Image.open(image_path)
    return np.asarray(img, dtype=np.uint8)

def plotMask(image_id):
    image_id = train_ids[0]

    plt.figure(figsize=(20, 20))
    img = get_image_matrix(get_car_image_files(image_id)[0])
    mask = get_image_matrix(get_car_image_files(image_id, True)[0])
    img_masked = cv2.bitwise_and(img, img, mask=mask)

    print("Image shape: {} | image type: {} | mask shape: {} | mask type: {}".format(img.shape, img.dtype, mask.shape, mask.dtype) )

    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(mask)
    plt.subplot(133)
    plt.imshow(img_masked)
    plt.show()

