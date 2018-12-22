from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.layers import Input

from shared.utils import setup_trainable_layers


def InceptionV3WithCustomLayers(nb_classes, input_shape, fc_size):
    """
    Adding custom final layers on Inception_V3 model with imagenet weights

    Args:
      nb_classes: # of classes
      input_shape: input shape of the images
    Returns:
      new keras model with new added last layer/s and the base model which new layers are added
    """
    base_model = InceptionV3(input_tensor=Input(shape=input_shape),
                             weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(fc_size * 2, activation='relu')(x)  # new FC layer, random init
    x = Dropout(0.3)(x)
    # x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
    # x = Dropout(0.5)(x)
    # x = Dense(FC_SIZE * 4, activation='relu')(x)  # new FC layer, random init
    # x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(outputs=predictions, inputs=base_model.input)
    return model, base_model


def build_finetuned_model(args, input_shape, fc_size):
    """
    Builds a finetuned InceptionV3 model from tensorflow implementation
    with imagenet weights loaded and setting up new fresh prediction layers at last

    Args:
        args: necessary args needed for training like train_data_dir, batch_size etc...
        input_shape: shape of input tensor
        fc_size: number of nodes to be used in last layers will be based on this value i.e its multiples may be used

    Returns:
        finetuned inceptionV3 model
    """
    # setup model
    iv3, base_iv3 = InceptionV3WithCustomLayers(args.nb_classes, input_shape, fc_size)
    setup_trainable_layers(iv3, args.layers_to_freeze)
    # compiling the model
    iv3.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return iv3
