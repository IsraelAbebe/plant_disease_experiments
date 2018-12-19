import tensorflow as tf
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D,Dropout
from tensorflow.python.keras.layers import Input

# This try except import is used only to support intellij IDE(pycharm)
# The except block import is what really works
try:
    from .utils import train_model, setup_args, INPUT_SHAPE, FC_SIZE
except:
    from utils import train_model, setup_args, INPUT_SHAPE, FC_SIZE


def InceptionV3WithCustomLayers(nb_classes, input_shape):
    """
    Adding custom final layers on Inception_V3 model with imagenet weights

    Args:
      nb_classes: # of classes
      input_shape: input shape of the images
    Returns:
      new keras model with new added last layer/s
    """
    base_model = InceptionV3(input_tensor=Input(shape=input_shape),
                             weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE * 2, activation='relu')(x)  # new FC layer, random init
    x = Dropout(0.3)(x)
    # x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
    # x = Dropout(0.5)(x)
    #x = Dense(FC_SIZE * 4, activation='relu')(x)  # new FC layer, random init
    #x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(outputs=predictions, inputs=base_model.input)
    return model


def setup_trainable_layers(model, layers_to_freeze=None):
    """
    Freeze the bottom layers and make trainable the remaining top layers.

    Args:
      model: keras model
      layers_to_freeze: number of layers to freeze or None to use the model as it is
    """
    if layers_to_freeze is not None:
        for layer in model.layers[:layers_to_freeze]:
            layer.trainable = False
        for layer in model.layers[layers_to_freeze:]:
            layer.trainable = True


def build_baseline_model(args):
    """
    Builds a finetuned InceptionV3 model from tensorflow implementation
    with imagenet weights loaded and setting up new fresh prediction layers at last

    Args:
        args: necessary args needed for training like train_data_dir, batch_size etc...

    Returns:
        finetuned inceptionV3 model
    """
    # setup model
    iv3 = InceptionV3WithCustomLayers(args.nb_classes, input_shape=INPUT_SHAPE)
    # setup layers to be trained or not
    setup_trainable_layers(iv3)
    # compiling the model
    iv3.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return iv3


if __name__ == "__main__":
    args = setup_args()
    iv3 = build_baseline_model(args)
    train_model(iv3, args)
