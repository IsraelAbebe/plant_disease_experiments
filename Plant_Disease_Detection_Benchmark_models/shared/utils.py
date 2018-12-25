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

IM_WIDTH, IM_HEIGHT = 100, 100  # fixed size for Inception_V3, N.B. 64 x 64 image shape was used for ResNet
INPUT_SHAPE = (IM_WIDTH, IM_HEIGHT, 3)
NB_EPOCHS = 50
BATCH_SIZE = 64
FC_SIZE = 512
NB_IV3_LAYERS_TO_FREEZE = 172

TRAIN_DIR = "../../../israels/plant-disease-experiments/Plant_Disease_Detection_Benchmark_models/dataset/segmentedspecies/train"  # "../../../Dataset/segmented_/train"
VAL_DIR = "../../../israels/plant-disease-experiments/Plant_Disease_Detection_Benchmark_models/dataset/segmentedspecies/val"  # "../../../Dataset/segmented_/val"

# model log and storage file details
MODEL_STORE_TEMPLATE = '{}-{}.h5'
MODEL_STORE_FOLDER = 'Models'
MODEL_LOG_TEMPLATE = '{}_{}_log.csv'
MODEL_LOG_FOLDER = 'logs'

# model type names
VGG_ARCHITECTURE = 'vgg'
INCEPTIONV3_ARCHITECTURE = 'inceptionv3'
RESNET_ARCHITECTURE = 'resnet'
SUPPORTED_MODEL_TYPES = {VGG_ARCHITECTURE, INCEPTIONV3_ARCHITECTURE, RESNET_ARCHITECTURE}

# mode mode names
CUSTOM = 'custom'  # custom implementation of the model architecture
BASELINE = 'baseline'  # standard model implementation from a library with no weights
FINETUNE = 'finetune'  # finetuned model of standard implementation
SUPPORTED_MODEL_MODES = {CUSTOM, BASELINE, FINETUNE}

AUGMENTATION_KWARGS = {
    'zoom_range': 2,
    'rotation_range': 0.1,
    'horizontal_flip': True
}


def get_cmd_args():
    """
    Get command line arguments if found or use default ones

    Returns:
         list of command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('model_identifier', help='A model name to identify model log and storage')
    parser.add_argument('model_type', choices=SUPPORTED_MODEL_TYPES, default=INCEPTIONV3_ARCHITECTURE,
                        help='Type of model to be used')
    parser.add_argument('model_mode', choices=SUPPORTED_MODEL_MODES, default=FINETUNE,
                        help='Mode of model implementation to be used')
    parser.add_argument('--train_dir', default=TRAIN_DIR, help='Directory containing training dataset')
    parser.add_argument('--val_dir', default=VAL_DIR, help='Directory containing validation dataset')
    parser.add_argument('--epochs', default=NB_EPOCHS, type=int)
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
    parser.add_argument('--layers_to_freeze', default=0, type=int,
                        help='Number of layers to freeze when finetuning a model')
    parser.add_argument('--augment', default=False, type=bool, help='Wheter to appply augmentation or not on dataset')
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


def get_data_generators(args, augment_kwargs):
    """
    Get training data and validation data generators

    Args:
        args: necessary arguments containing train_data_dir, val_data_dir, batch_size etc...
        augment_kwargs: data augmentation specifications dict to be used as kwargs

    Returns:
        tuple of training and validation data generators
    """
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, **augment_kwargs)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, **augment_kwargs)

    train_data_generator = train_datagen.flow_from_directory(args.train_dir, target_size=(IM_WIDTH, IM_HEIGHT),
                                                             batch_size=args.batch_size)
    val_data_generator = val_datagen.flow_from_directory(args.val_dir, target_size=(IM_WIDTH, IM_HEIGHT),
                                                         batch_size=args.batch_size)

    return train_data_generator, val_data_generator


def setup_trainable_layers(model, layers_to_freeze=None):
    """
    Freeze the bottom layers and make trainable the remaining top layers.

    Args:
      model: keras model
      layers_to_freeze: number of layers to freeze or None to use the model as it is
    """
    if layers_to_freeze is not None:
        if not (0 <= layers_to_freeze <= len(model.layers)):
            raise ValueError('`layers_to_freeze` argument must be between 0 and {}'.format(len(model.layers)))

        for layer in model.layers[:layers_to_freeze]:
            layer.trainable = False
        for layer in model.layers[layers_to_freeze:]:
            layer.trainable = True


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

    # augment dataset if required or is VGG model
    if args.model_type == VGG_ARCHITECTURE or args.augment:
        augment_kwargs = AUGMENTATION_KWARGS
    else:
        augment_kwargs = {}

    identifier = args.model_identifier
    csv_log_filename = get_model_log_name(args.model_type, identifier)

    nb_train_samples = get_nb_files(args.train_dir)
    nb_val_samples = get_nb_files(args.val_dir)
    output_model_file = get_model_storage_name(args.model_type, identifier)

    train_data_generator, val_data_generator = get_data_generators(args, augment_kwargs)

    # set up callbacks to be used for model training
    callbacks = [ CSVLogger(csv_log_filename) ]
    if args.model_type != RESNET_ARCHITECTURE:
        callbacks.extend([
            ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6),
            EarlyStopping(min_delta=0.001, patience=15, verbose=5)
        ])

    # setup additional kwargs if the model needs it
    if args.model_type in {VGG_ARCHITECTURE, RESNET_ARCHITECTURE}:
        additional_kwargs = {
            'class_weight': 'auto'
        }
    else:
        additional_kwargs = {}

    # train the model using data generator
    history_ft = model.fit_generator(
        train_data_generator, epochs=args.epochs, callbacks=callbacks,
        steps_per_epoch=nb_train_samples // args.batch_size,
        validation_data=val_data_generator,
        validation_steps=nb_val_samples // args.batch_size, **additional_kwargs
    )

    # save the model
    model.save(output_model_file)

    # plot history if needed
    if plot:
        plot_training(history_ft)


def get_model_storage_name(model_type, identifier):
    """
    Get storage path used for saving a model and create its folder if required

    Args:
        model_type: type of the model architecture like VGG, InceptionV3
        identifier: a partial string that is supposed to make the filename unique from other models

    Returns:
        relative filepath used to save the model
    """
    if not os.path.isdir(MODEL_STORE_FOLDER):
        os.mkdir(MODEL_STORE_FOLDER)
    return os.path.join(MODEL_STORE_FOLDER, MODEL_STORE_TEMPLATE).format(model_type, identifier)


def get_model_log_name(model_type, identifier):
    """
    Get filename used to log a model's epochs and create its folder if required

    Args:
        model_type: type of the model architecture like VGG, InceptionV3
        identifier:a partial string that is supposed to make the filename unique from other models

    Returns:
        filename used to save model's log
    """
    if not os.path.isdir(MODEL_LOG_FOLDER):
        os.mkdir(MODEL_LOG_FOLDER)
    return os.path.join(MODEL_LOG_FOLDER, MODEL_LOG_TEMPLATE).format(model_type, identifier)


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
