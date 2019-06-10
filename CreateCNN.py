from keras.models import Sequential
classifier = Sequential()

from keras.layers import Conv2D
classifier.add(Conv2D(filters=32, kernel_size=(3,3),strides=(1, 1), input_shape=(64,64,3), activation='relu'))

from keras.layers import MaxPooling2D
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

from keras.layers import Flatten
classifier.add(Flatten())

from keras.layers import Dense
from keras.layers import Dropout
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units=3, activation='softmax'))

classifier.compile( optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
# applying transformation to image
train_datagen = ImageDataGenerator(
 rescale=1./255,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:\\Users\\x97272\\newdata_set\\training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
test_set = test_datagen.flow_from_directory(
        'C:\\Users\\x97272\\newdata_set\\test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

from IPython.display import display 

classifier.fit_generator(
        training_set,
        steps_per_epoch=250,
        epochs=10,
        validation_data=test_set, validation_steps=150)

#------save model-------#

from keras.models import load_model

classifier.save('modelv4.h5')
