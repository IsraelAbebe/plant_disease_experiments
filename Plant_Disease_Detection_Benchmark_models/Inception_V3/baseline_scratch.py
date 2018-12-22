from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.layers import Input


def build_baseline_model(args, input_shape):
    """
    Builds a baseline InceptionV3 model from tensorflow implementation
    with no trained weights loaded and including top layers for prediction

    Args:
        args: necessary args needed for training like train_data_dir, batch_size etc...
        input_shape: shape of input tensor

    Returns:
        baseline inceptionV3 model
    """
    iv3 = InceptionV3(input_tensor=Input(shape=input_shape), weights=None,
                      include_top=True, classes=args.nb_classes)
    iv3.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return iv3
