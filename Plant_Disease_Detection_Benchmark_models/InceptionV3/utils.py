import os
import glob
import numpy as np

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IM_WIDTH, IM_HEIGHT = 100, 100  # fixed size for InceptionV3
INPUT_SHAPE = (IM_WIDTH, IM_HEIGHT, 3)
NB_EPOCHS = 50
BATCH_SIZE = 64
FC_SIZE = 512
NB_IV3_LAYERS_TO_FREEZE = 172

TRAIN_DIR = "../../../Dataset/segmented_/train"
VAL_DIR = "../../../Dataset/segmented_/val"

MODEL_STORE_TEMPLATE = "../Models/InceptionV3-{}.h5"
MODEL_LOG_TEMPLATE = "{}_iv3_log.csv"

nb_classes = len(glob.glob(TRAIN_DIR + "/*"))

def train_model(model, plot=False):
    identifier = input('Enter model name to identify log and saved model: ')
    CSV_LOG_FILE = MODEL_LOG_TEMPLATE.format(identifier)

    nb_train_samples = get_nb_files(TRAIN_DIR)
    nb_val_samples = get_nb_files(VAL_DIR)
    nb_epoch = int(NB_EPOCHS)
    batch_size = int(BATCH_SIZE)
    output_model_file = MODEL_STORE_TEMPLATE.format(identifier)

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(IM_WIDTH, IM_HEIGHT),
                                                        batch_size=batch_size)
    validation_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(IM_WIDTH, IM_HEIGHT),
                                                            batch_size=batch_size)

    callbacks = [
        ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6),
        EarlyStopping(min_delta=0.001, patience=10),
        CSVLogger(CSV_LOG_FILE)
    ]
    history_ft = model.fit_generator(train_generator, epochs=nb_epoch, steps_per_epoch=nb_train_samples // batch_size,
                                   validation_data=validation_generator, use_multiprocessing=True,
                                   validation_steps=nb_val_samples // batch_size,
                                   callbacks=callbacks)

    model.save(output_model_file)

    if plot:
        plot_training(history_ft)


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
