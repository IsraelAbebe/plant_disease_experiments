# N.B. using keras rather than tensorflow.keras implementation
#      since vggface uses keras so to be compatible with it
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import Input
from keras.optimizers import SGD

from keras_vggface.vggface import VGGFace

from shared.utils import setup_trainable_layers


def VGGWithCustomLayers(nb_classes, input_shape, fc_size):
    """
    Adding custom final layers on VGG model with no weights
    Args:
      nb_classes: # of classes
      input_shape: input shape of the images
      fc_size: number of nodes to be used in last layers will be based on this value i.e its multiples may be used
    Returns:
      new keras model with new added last layer/s and the base model which new layers are added
    """
    # setup model
    base_model = VGGFace(include_top=False, input_tensor=Input(shape=input_shape))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(fc_size, activation='relu')(x)  # new FC layer, random init
    x = Dense(fc_size * 2, activation='relu')(x)  # new FC layer, random init
    x = Dense(fc_size * 4, activation='relu')(x)  # new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(outputs=predictions, inputs=base_model.input)
    return model, base_model


def build_finetuned_model(args, input_shape, fc_size):
    """
    Builds a finetuned VGG model from VGGFace implementation
    with no weights loaded and setting up new fresh prediction layers at last
    Args:
        args: necessary args needed for training like train_data_dir, batch_size etc...
        input_shape: shape of input tensor
        fc_size: number of nodes to be used in last layers will be based on this value i.e its multiples may be used
    Returns:
        finetuned vgg model
    """
    # setup model
    vgg, base_vgg = VGGWithCustomLayers(args.nb_classes, input_shape, fc_size)
    # setup layers to be trained or not
    setup_trainable_layers(vgg, args.layers_to_freeze)
    # compiling the model
    vgg.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    return vg