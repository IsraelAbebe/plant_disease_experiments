import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from keras.layers import Input

print('tf version: ', tf.__version__)

try:
    from .utils import train_model, nb_classes, INPUT_SHAPE, FC_SIZE
except:
    from utils import train_model, nb_classes, INPUT_SHAPE, FC_SIZE


def InceptionV3WithCustomLayers(input_shape, nb_classes):
    """
    Adding custom final layers on InceptionV3 model with imagenet weights

    Args:
      input_shape: input shape of the images
      nb_classes: # of classes
    Returns:
      new keras model with last layer/s
    """
    base_model = InceptionV3(input_tensor=Input(shape=input_shape),
                             weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE * 2, activation='relu')(x)  # new FC layer, random init
    x = Dropout(0.5)(x)
    # x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
    # x = Dropout(0.5)(x)
    #x = Dense(FC_SIZE * 4, activation='relu')(x)  # new FC layer, random init
    #x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(output=predictions, input=base_model.input)
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


# setup model
model = InceptionV3WithCustomLayers(INPUT_SHAPE, nb_classes)
# setup layers to be trained or not
setup_trainable_layers(model)
# compiling the model
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_model(model)
