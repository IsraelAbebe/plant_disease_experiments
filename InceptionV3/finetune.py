import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.layers import Input

IM_WIDTH, IM_HEIGHT = 100, 100  # fixed size for InceptionV3
NB_EPOCHS = 50
BAT_SIZE = 64
FC_SIZE = 512
NB_IV3_LAYERS_TO_FREEZE = 172


# python finetune.py --train_dir "../dataset/color/train" --val_dir "../dataset/color/val"


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('Inception_Finetuning_log.csv')



def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt




def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
      base_model: keras model excluding top
      nb_classes: # of classes
    Returns:
      new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
    x = Dropout(0.5)(x)
    x = Dense(FC_SIZE * 2, activation='relu')(x)  # new FC layer, random init
    x = Dropout(0.5)(x)
    #x = Dense(FC_SIZE * 4, activation='relu')(x)  # new FC layer, random init
    #x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(output=predictions, input=base_model.input)
    return model


def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
      model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


def train(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)

    # data prep
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(args.train_dir, target_size=(IM_WIDTH, IM_HEIGHT),
                                                        batch_size=batch_size)

    validation_generator = test_datagen.flow_from_directory(args.val_dir, target_size=(IM_WIDTH, IM_HEIGHT),
                                                            batch_size=batch_size)

    # setup model
    base_model = InceptionV3(input_tensor=Input(shape=(IM_HEIGHT, IM_WIDTH, 3)), weights='imagenet', include_top=False)

    model = add_new_last_layer(base_model, nb_classes)

    
    # fine-tuning
    setup_to_finetune(model)

    history_ft = model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=nb_epoch,
                                     validation_data=validation_generator,
                                     validation_steps=nb_val_samples // batch_size,callbacks=[lr_reducer,early_stopper,csv_logger])

    model.save(args.output_model_file)

    plot_training(history_ft)


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


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir")
    a.add_argument("--val_dir")
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--output_model_file", default="../Models/InceptionV3-finetuning.h5")
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

train(args)
