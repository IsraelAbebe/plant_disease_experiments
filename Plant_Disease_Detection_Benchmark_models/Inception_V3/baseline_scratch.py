from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.layers import Input

# shameful hack to support running this script with no error
# it just includes this script parent folder in sys.path(what a shame)
if __name__ == "__main__" :
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))

# This try except import is used only to support intellij IDE(pycharm)
# The except block import is what really works with the support of the above shameful hack
try:
    from Plant_Disease_Detection_Benchmark_models.utils import train_model, setup_args, INPUT_SHAPE, INCEPTIONV3_ARCHITECTURE
except:
    from utils import train_model, setup_args, INPUT_SHAPE, INCEPTIONV3_ARCHITECTURE


def build_baseline_model(args):
    """
    Builds a baseline InceptionV3 model from tensorflow implementation
    with no trained weights loaded and including top layers for prediction

    Args:
        args: necessary args needed for training like train_data_dir, batch_size etc...

    Returns:
        baseline inceptionV3 model
    """
    iv3 = InceptionV3(input_tensor=Input(shape=INPUT_SHAPE), weights=None,
                      include_top=True, classes=args.nb_classes)
    iv3.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return iv3


if __name__ == "__main__":
    args = setup_args()
    iv3 = build_baseline_model(args)
    train_model(iv3, args, INCEPTIONV3_ARCHITECTURE)
