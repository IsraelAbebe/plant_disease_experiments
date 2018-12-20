from tensorflow.python._pywrap_tensorflow_internal import Flatten
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout

# shameful hack to support running this script with no error
# it just includes this script parent folder in sys.path(what a shame)
if __name__ == "__main__":
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))

# This try except import is used only to support intellij IDE(pycharm)
# The except block import is what really works with the support of the above shameful hack
try:
    from Plant_Disease_Detection_Benchmark_models.utils import train_model, setup_args, INPUT_SHAPE, VGG_ARCHITECTURE
except:
    from utils import train_model, setup_args, INPUT_SHAPE, VGG_ARCHITECTURE


def VGG(nb_classes, input_shape):
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    '''
        Trained on 4 GPUs for 2–3 weeks       :P

    '''

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, activation='softmax'))

    return model


def build_custom_model(args):
    """
    Builds a baseline VGG model based on the paper
    with no trained weights loaded

    Args:
        args: necessary args needed for training like train_data_dir, batch_size etc...

    Returns:
        baseline vgg model
    """
    model = VGG(args.nb_classes, input_shape=INPUT_SHAPE)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    args = setup_args()
    iv3 = build_custom_model(args)
    train_model(iv3, args, VGG_ARCHITECTURE)
