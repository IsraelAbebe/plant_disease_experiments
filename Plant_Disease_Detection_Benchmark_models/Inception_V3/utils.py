import os
import glob
import numpy as np
import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from tensorflow.python.keras.applications.inception_v3 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import Callback

IM_WIDTH, IM_HEIGHT = 100, 100  # fixed size for Inception_V3
INPUT_SHAPE = (IM_WIDTH, IM_HEIGHT, 3)
NB_EPOCHS = 50
BATCH_SIZE = 64
FC_SIZE = 512
NB_IV3_LAYERS_TO_FREEZE = 172

TRAIN_DIR = "../../../../israels/plant-disease-experiments/Plant_Disease_Detection_Benchmark_models/dataset/segmentedspecies/train"  # "../../../Dataset/segmented_/train"
VAL_DIR = "../../../../israels/plant-disease-experiments/Plant_Disease_Detection_Benchmark_models/dataset/segmentedspecies/val"  # "../../../Dataset/segmented_/val"

MODEL_STORE_TEMPLATE = "../Models/Inception_V3-{}.h5"
MODEL_LOG_TEMPLATE = "{}_iv3_log.csv"


def get_cmd_args():
    """
    Get command line arguments if found or use default ones

    Returns:
         list of command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='A model name to identify model log and storage')
    parser.add_argument('--train_dir', default=TRAIN_DIR)
    parser.add_argument('--val_dir', default=VAL_DIR)
    parser.add_argument('--epochs', default=NB_EPOCHS, type=int)
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
    parser.add_argument('--layers_to_freeze', default=NB_IV3_LAYERS_TO_FREEZE, type=int)
    args = parser.parse_args()

    return args


def setup_args():
    """
    Get cmd args and add other necessary args

    Returns:
         list of necessary args needed for training a model
    """
    args = get_cmd_args()

    # N.B. assumes training directory contains only folders for each class
    args.nb_classes = len(glob.glob(args.train_dir + "/*"))

    return args


def get_data_generators(args):
    """
    Get training data and validation data generators

    Args:
        args: necessary arguments containing train_data_dir, val_data_dir, batch_size etc...

    Returns:
        tuple of training and validation data generators
    """
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_data_generator = train_datagen.flow_from_directory(args.train_dir, target_size=(IM_WIDTH, IM_HEIGHT),
                                                             batch_size=args.batch_size)
    val_data_generator = val_datagen.flow_from_directory(args.val_dir, target_size=(IM_WIDTH, IM_HEIGHT),
                                                         batch_size=args.batch_size)

    return train_data_generator, val_data_generator


def train_model(model, args, plot=False):
    """
    Trains a model, logs on every epoch, saves the model when it finishes
    and plot the training history if required

    Args:
        model: the model which will be trained
        args: necessary args needed for training like train_data_dir, batch_size etc...
        plot: whether to plot the training history or not
    """
    print('tf version: ', tf.__version__)

    identifier = args.model_name
    csv_log_filename = get_model_log_name(identifier)

    nb_train_samples = get_nb_files(args.train_dir)
    nb_val_samples = get_nb_files(args.val_dir)
    output_model_file = get_model_storage_name(identifier)

    train_data_generator, val_data_generator = get_data_generators(args)

    callbacks = [
        ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6),
        # MonitoringCallback(),
        # CustomEarlyStopping(min_delta=0.001, patience=10),
        EarlyStopping(min_delta=0.001, patience=15, verbose=5),
        CSVLogger(csv_log_filename)
    ]

    history_ft = model.fit_generator(
        train_data_generator, epochs=args.epochs, callbacks=callbacks,
        steps_per_epoch=nb_train_samples // args.batch_size,
        validation_data=val_data_generator,
        validation_steps=nb_val_samples // args.batch_size,
    )

    model.save(output_model_file)

    if plot:
        plot_training(history_ft)


def get_model_storage_name(identifier):
    """
    Get storage path used for saving a model

    Args:
        identifier: a partial string that is supposed to make the filename unique from other models

    Returns:
        relative filepath used to save the model
    """
    return MODEL_STORE_TEMPLATE.format(identifier)


def get_model_log_name(identifier):
    """
    Get filename used to log a model's epochs

    Args:
        identifier:a partial string that is supposed to make the filename unique from other models

    Returns:
        filename used to save model's log
    """
    return MODEL_LOG_TEMPLATE.format(identifier)


def get_nb_files(directory):
    """
    Get number of files starting from subfolders by searching them recursively

    Args:
        directory: direcotry path to count files in its subfolders
    """
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def plot_training(history):
    """
    Plots training history

    Args:
        history: dict returned from tf model training
    """
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


class MonitoringCallback(Callback):
    """
    N.B. Debugging code. Not tested.
    Prints a bunch of needed information on every epoch end
    """

    def __init__(self):
        super(MonitoringCallback, self).__init__()

        self.past_val_loss = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] is None:
            print("Lost val_loss itself...weird")
            return

        print(
            "\nLast val_loss difference: {}\n"
                .format(logs['val_loss'] - self.past_val_loss)
        )
        self.past_val_loss = logs['val_loss']


class CustomEarlyStopping(Callback):
    """
    N.B. Debugging code. Not tested.
    Early stops training other way than library's earlystopping mechnanism
    and prints a bunch of needed information on every epoch end
    """

    def __init__(self, monitor='val_loss', min_delta=0, patience=0):
        super(CustomEarlyStopping, self).__init__()

        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.last_fezaza_epoch = 0
        self.last_monitored_value = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs[self.monitor] is None:
            print("Lost {} itself...weird".format(self.monitor))
            return

        monitored_value_diff = logs[self.monitor] - self.last_monitored_value
        epoch_diff = epoch - self.last_fezaza_epoch

        if abs(monitored_value_diff) > self.min_delta:
            self.last_fezaza_epoch = epoch
            print('\nfezaza epoch updated to ', self.last_fezaza_epoch)

        print('\n{} diff: {}, epoch diff: {}\n'.format(self.monitor, monitored_value_diff, epoch_diff))
        if epoch_diff >= self.patience:
            print('Custom early stopping at epoch ', epoch)
            self.model.stop_training = True
