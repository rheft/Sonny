import sys
sys.path.insert(0, '/Users/rheft/dev/Sonny/')
from config import data_locale
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import PIL
#### Building CNN

# Initialize the CNN
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(16, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Add second convolutional/pooled layers
# classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(activation = 'relu', units = 64))
classifier.add(Dense(activation = 'sigmoid', units = 1))

# Compile CNN!!! for more than 1 category: the loss funcitn is categorical_crossentropy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

## Fit the image to the
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        data_locale+'cnn_datasets/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        data_locale+'cnn_datasets/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=(8000/32),
        epochs=25,
        validation_data=test_set,
        validation_steps=800)
