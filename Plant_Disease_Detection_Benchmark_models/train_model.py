from shared.utils import *

import Inception_V3
import VGG
import ResNet


def get_model(args, input_shape):
    """
    Checks validity of requested model and gets appropriate model based on provided model type and model mode

    Args:
        args: necessary args needed for training like train_data_dir, batch_size etc...
            args have to also include below variables
            model_type: type of the model like VGG, InceptionV3 ...
            model_mode: model mode of implementation like customly implemented model, model from library, finetuned model implementation...
        input_shape: shape of input tensor

    Returns:
        keras model
    """
    if args.model_type == INCEPTIONV3_ARCHITECTURE:
        if args.model_mode == FINETUNE:
            model = Inception_V3.build_finetuned_model(args, input_shape, FC_SIZE)
        elif args.model_mode == CUSTOM:
            model = Inception_V3.build_custom_model(args, input_shape)
        elif args.model_mode == BASELINE:
            model = Inception_V3.build_baseline_model(args, input_shape)
        else:
            raise ValueError('Unsuppported model mode `{}` provided: Supported modes are {}'.format(args.model_mode,
                                                                                                    SUPPORTED_MODEL_MODES))
    elif args.model_type == VGG_ARCHITECTURE:
        if args.model_mode == FINETUNE:
            model = VGG.build_finetuned_model(args, input_shape, FC_SIZE)
        elif args.model_mode == CUSTOM:
            model = VGG.build_custom_model(args, input_shape)
        elif args.model_mode == BASELINE:
            raise NotImplementedError('Baseline mode for VGG architecture is not implemented yet. Choose another mode')
        else:
            raise ValueError('Unsuppported model mode `{}` provided: Supported modes are {}'.format(args.model_mode,
                                                                                                    SUPPORTED_MODEL_MODES))
    elif args.model_type == RESNET_ARCHITECTURE:
        if args.model_mode == FINETUNE:
            raise NotImplementedError('Finetune mode for ResNet architecture is not implemented yet. Choose another mode')
        elif args.model_mode == CUSTOM:
            # N.B. originally used input shape was 64 x 64...in case something went wrong with current default input shape provided
            model = ResNet.build_custom_model(args, input_shape)
        elif args.model_mode == BASELINE:
            raise NotImplementedError('Baseline mode for ResNet architecture is not implemented yet. Choose another mode')
        else:
            raise ValueError('Unsuppported model mode `{}` provided: Supported modes are {}'.format(args.model_mode,
                                                                                                    SUPPORTED_MODEL_MODES))
    else:
        raise ValueError(
            'Unsuppported model type `{}` provided: Supported model types are are {}'.format(args.model_type,
                                                                                             SUPPORTED_MODEL_TYPES))

    return model


def main():
    """
    Main code to be executed when this file is run as script
    """
    args = setup_args()
    model = get_model(args, INPUT_SHAPE)
    train_model(model, args)


if __name__ == '__main__':
    main()
