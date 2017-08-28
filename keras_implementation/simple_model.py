from keras.layers import Conv2D
from keras.models import Sequential

# Create simple model

def model():

	model = Sequential()
	model.add( Conv2D(16, 3, activation='relu', padding='same', input_shape=(320, 480, 12) ) )
	model.add( Conv2D(32, 3, activation='relu', padding='same') )
	model.add( Conv2D(1, 5, activation='sigmoid', padding='same') )
		
	return model
