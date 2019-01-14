from keras.layers import Flatten
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout

try:
    from shared.utils import train_model, setup_args, INPUT_SHAPE, VGG_ARCHITECTURE
except:
    from shared import train_model, setup_args, INPUT_SHAPE, VGG_ARCHITECTURE


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


def build_custom_model(args, input_shape):
    """
    Builds a baseline VGG model based on the paper
    with no trained weights loaded

    Args:
        args: necessary args needed for training like train_data_dir, batch_size etc...
        input_shape: shape of input tensor

    Returns:
        baseline vgg model
    """
    model = VGG(args.nb_classes, input_shape=input_shape)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
