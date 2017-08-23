#Importing the files
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers

#Loading the data
img_width, img_height = 150, 150
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

#BUILDING THE MODEL
model = Sequential() #Linear flow of data
model.add(Convolution2D(32, 3, 3, input_shape=(img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Adding next layer
model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Adding next layer
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Adding the fully connected components
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])

#TRAINING
nb_epochs = 5
nb_train_samples = 1600
nb_validation_samples = 800
model.fit_generator(train_generator,
                   samples_per_epoch=nb_train_samples,
                   nb_epoch = nb_epochs,
                   validation_data = validation_generator,
                   nb_val_samples = nb_validation_samples)
model.save_weights('models/memecnn.h5')

model.evaluate_generator(validation_generator, nb_validation_samples)

print("img 1 = meme")
img = load_img('test/test_img_1.jpg')
img = img_to_array(img)
img = img.reshape((1,)+img.shape)
img = img * (1. / 255)
prediction = model.predict(img)
print(prediction)

print("img 2 = notmeme")
img = load_img('test/test_img_2.jpg')
img = img_to_array(img)
img = img.reshape((1,)+img.shape)
img = img * (1. / 255)
prediction = model.predict(img)
print(prediction)

print("img 3 =  meme ")
img = load_img('test/test_img_3.jpg')
img = img_to_array(img)
img = img.reshape((1,)+img.shape)
img = img * (1. / 255)
prediction = model.predict(img)
print(prediction)

print("img 4 =  meme ")
img = load_img('test/test_img_4.jpg')
img = img_to_array(img)
img = img.reshape((1,)+img.shape)
img = img * (1. / 255)
prediction = model.predict(img)
print(prediction)