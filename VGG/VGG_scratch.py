from __future__ import print_function
import os
import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
from keras import regularizers

train_dir = "../dataset/color/train"
test_dir = "../dataset/color/val"
csv_logger = CSVLogger('log.csv', append=True, separator=';')


def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


def custome_loss(y_pred, y_true):
    print(y_pred, y_true)
    return 0


batch_size = 128
epochs = 30
nb_train_samples = get_nb_files(train_dir)
num_classes = len(glob.glob(train_dir + "/*"))
nb_val_samples = get_nb_files(test_dir)

# input image dimensions
IM_WIDTH, IM_HEIGHT = 100, 100
input_shape = (IM_WIDTH, IM_HEIGHT, 3)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(IM_WIDTH, IM_HEIGHT), batch_size=batch_size)

test_generator = test_datagen.flow_from_directory(test_dir, target_size=(IM_WIDTH, IM_HEIGHT), batch_size=batch_size)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='valid'))
model.add(Dropout(0.5))
model.add(Conv2D(256, (3, 3), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='valid'))
model.add(Dropout(0.6))
model.add(Conv2D(512, (3, 3), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(1024, (3, 3), activation='relu', padding='valid'))
# model.add(Dropout(0.75))
# model.add(Conv2D(1024, (3, 3), activation='relu', padding='valid'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy'])
history_train = model.fit_generator(train_generator, nb_epoch=epochs, steps_per_epoch=nb_train_samples // batch_size,
                                    validation_data=test_generator, nb_val_samples=nb_val_samples // batch_size,
                                    class_weight='auto', callbacks=[csv_logger])
plot_training(history_train)

model.save("../Models/VGG_scratch.h5")
