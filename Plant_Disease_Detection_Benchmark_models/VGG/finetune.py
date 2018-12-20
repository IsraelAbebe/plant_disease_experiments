from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.optimizers import SGD
from keras_vggface.vggface import VGGFace

# shameful hack to support running this script with no error
# it just includes this script parent folder in sys.path(what a shame)

if __name__ == "__main__":
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))

# This try except import is used only to support intellij IDE(pycharm)
# The except block import is what really works with the support of the above shameful hack
try:
    from Plant_Disease_Detection_Benchmark_models.utils import train_model, setup_args, setup_trainable_layers, \
        INPUT_SHAPE, VGG_ARCHITECTURE, FC_SIZE
except:
    from utils import train_model, setup_args, setup_trainable_layers, INPUT_SHAPE, VGG_ARCHITECTURE, FC_SIZE


def VGGWithCustomLayers(nb_classes, input_shape):
    """
    Adding custom final layers on VGG model with no weights

    Args:
      nb_classes: # of classes
      input_shape: input shape of the images
    Returns:
      new keras model with new added last layer/s and the base model which new layers are added
    """
    # setup model
    base_model = VGGFace(include_top=False, input_tensor=Input(shape=input_shape))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
    x = Dense(FC_SIZE * 2, activation='relu')(x)  # new FC layer, random init
    x = Dense(FC_SIZE * 4, activation='relu')(x)  # new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(output=predictions, input=base_model.input)
    return model, base_model


def build_finetuned_model(args):
    """
    Builds a finetuned VGG model from VGGFace implementation
    with no weights loaded and setting up new fresh prediction layers at last

    Args:
        args: necessary args needed for training like train_data_dir, batch_size etc...

    Returns:
        finetuned vgg model
    """
    # setup model
    vgg, base_vgg = VGGWithCustomLayers(args.nb_classes, input_shape=INPUT_SHAPE)
    # setup layers to be trained or not
    setup_trainable_layers(vgg, args.layers_to_freeze)
    # compiling the model
    vgg.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    return vgg


if __name__ == "__main__":
    args = setup_args()
    vgg = build_finetuned_model(args)
    train_model(vgg, args, VGG_ARCHITECTURE)
