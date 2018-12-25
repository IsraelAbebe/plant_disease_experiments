import ResNet.resnet as resnet


def build_custom_model(args, input_shape):
    """
    Builds a custom ResNet model based on ResNet paper

    Args:
        args: necessary args needed for training like train_data_dir, batch_size etc...
        input_shape: shape of input tensor

    Returns:
        custom ResNet model
    """

    model = resnet.ResnetBuilder.build_resnet_18((3, input_shape[0], input_shape[1]), args.nb_classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
