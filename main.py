import os
import argparse
import subprocess
import numpy as np

from PIL import Image

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_v3 import preprocess_input

# species names
APPLE = 'apple'
BEAN = 'bean'
BLUEBERRY = 'blueberry'
CHERRY = 'cherry'
CORN = 'corn'
GRAPE = 'grape'
GRAPEFRUIT = 'grapefruit'
ORANGE = 'orange'
PEACH = 'peach'
PEPPER = 'pepper'
POTATO = 'potato'
RASPBERRY = 'raspberry'
SORGHUM = 'sorghum'
SOYBEAN = 'soybean'
SQUASH = 'squash'
STRAWBERRY = 'strawberry'
SUGARCANE = 'sugarcane'
TOMATO = 'tomato'

# all species and supported species names
SPECIES = [APPLE, BEAN, BLUEBERRY, CHERRY, CORN, GRAPE, GRAPEFRUIT, ORANGE, PEACH,
           PEPPER, POTATO, RASPBERRY, SORGHUM, SOYBEAN, SQUASH, STRAWBERRY, SUGARCANE, TOMATO]
DISEASE_SUPPORTED_SPECIES = {APPLE, CHERRY, CORN, GRAPE, PEACH, PEPPER, POTATO, STRAWBERRY, SUGARCANE, TOMATO, }

# classes for each species
APPLE_CLASSES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']
CHERRY_CLASSES = ['Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy']
CORN_CLASSES = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy']
GRAPE_CLASSES = ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                 'Grape___healthy']
PEACH_CLASSES = ['Peach___Bacterial_spot', 'Peach___healthy']
PEPPER_CLASSES = ['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy']
POTATO_CLASSES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
STRAWBERRY_CLASSES = ['Strawberry___Leaf_scorch', 'Strawberry___healthy']
SUGARCANE_CLASSES = ['Sugarcane leaf spot', 'Sugarcane aphid', 'Sugarcane coal fouling']
TOMATO_CLASSES = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                  'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                  'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                  'Tomato___healthy']

# all species classes with their species name as key
PLANT_CLASSES = {
    APPLE: APPLE_CLASSES,
    CHERRY: CHERRY_CLASSES,
    CORN: CORN_CLASSES,
    GRAPE: GRAPE_CLASSES,
    PEACH: PEACH_CLASSES,
    PEPPER: PEPPER_CLASSES,
    POTATO: POTATO_CLASSES,
    STRAWBERRY: STRAWBERRY_CLASSES,
    SUGARCANE: SUGARCANE_CLASSES,
    TOMATO: TOMATO_CLASSES,
}

# types of models to be used for predictions
VGG_ARCHITECTURE = 'vgg'
INCEPTIONV3_ARCHITECTURE = 'inceptionv3'
SUPPORTED_MODEL_TYPES = {VGG_ARCHITECTURE, INCEPTIONV3_ARCHITECTURE}

# modes of detection i.e detecting plant disease or species
DISEASE_DETECTION = 'disease_detection'
SPECIES_DETECTION = 'species_detection'

# image target sizes for our supported model architectures
TARGET_IMAGE_SIZES = {
    VGG_ARCHITECTURE: {
        DISEASE_DETECTION: (64, 64),
        SPECIES_DETECTION: (100, 100),
    },
    INCEPTIONV3_ARCHITECTURE: {
        DISEASE_DETECTION: (100, 100),
        SPECIES_DETECTION: (100, 100),
    }
}

# vgg models to be used with their species name as key
VGG_MODELS = {
    APPLE: 'Apple_0.9395_VGG.h5',
    CHERRY: 'Cherry_0.9873_VGG.h5',
    CORN: 'Corn_0.8926_VGG.h5',
    GRAPE: 'Grape_0.9293_VGG.h5',
    PEACH: 'Peach_97_VGG.h5',
    TOMATO: 'Tomato_0.8675_VGG.h5',
    PEPPER: 'pepper_95.90.h5',
    POTATO: 'potato_90.62.h5',
    STRAWBERRY: 'starwberry_99.h5',
    SUGARCANE: 'Sugarcane_0.8356_VGG.h5'
}

# inceptionv3 models to be used with their species name as key
INCEPTIONV3_MODELS = {
    APPLE: 'InceptionV3-scratch_segApple.h5',
    CHERRY: 'InceptionV3-scratch_segCherry.h5',
    CORN: 'InceptionV3-scratch_segCorn.h5',
    GRAPE: 'InceptionV3-scratch_segGrape.h5',
    PEACH: 'InceptionV3-scratch_segPeach.h5',
    TOMATO: 'InceptionV3-scratch_segTomato.h5',
    PEPPER: 'InceptionV3-scratch_segPepper.h5',
    POTATO: 'InceptionV3-scratch_segPotato.h5',
    STRAWBERRY: 'InceptionV3-scratch_segStrawberry.h5',
    SUGARCANE: 'InceptionV3-scratch_segSugarcane.h5'
}

# base path from where models will be loaded
MODEL_STORAGE_BASE = 'Plant_Disease_Detection_Benchmark_models/Models'


def get_classes(species_name):
    """
    Get classes of disease for a species

    Args:
        species_name: name of species

    Returns:
        a list of disease classes for a specific species
    """
    return PLANT_CLASSES[species_name]


def get_disease_model(species, model_type):
    """
    Get appropriate disease classifier model file name

    Args:
        species: species name to identify which species model should be used
        model_type: type of model to be used for prediction

    Returns:
        disease classifier model file name
    """
    if species not in DISEASE_SUPPORTED_SPECIES:
        raise ValueError("`{}` species has no disease model yet.\n"
                         "Species tha have disease models are {}".format(species, DISEASE_SUPPORTED_SPECIES))

    if model_type == VGG_ARCHITECTURE:
        return VGG_MODELS[species]
    elif model_type == INCEPTIONV3_ARCHITECTURE:
        return INCEPTIONV3_MODELS[species]
    else:
        raise ValueError("No such `{}` model type is supported.\n"
                         "Supported model types are {}".format(model_type, SUPPORTED_MODEL_TYPES))


def get_species_model(model_type):
    """
    Get appropriate species classifier model file name

    Args:
        species: species name to identify which species model should be used
        model_type: type of model to be used for prediction

    Returns:
        species classifier model file name
    """

    if model_type == VGG_ARCHITECTURE:
        return 'VGG_all_100p_94.h5'
    elif model_type == INCEPTIONV3_ARCHITECTURE:
        return 'InceptionV3-scratch_segspecies.h5'
    else:
        raise ValueError("No such `{}` model type is supported.\n"
                         "Supported model types are {}".format(model_type, SUPPORTED_MODEL_TYPES))


def get_predictions(model_path, img_path, img_target_size):
    """
    Loads model and image and make predictions using them

    Args:
        model_path: filesystem path of model
        img_path: filesystem path of image
        img_target_size: target image size to reshape the image if necessary

    Returns:
        a tuple of:
            1. array of prediction values by the model for all classes
            2. array of indices that can sort the classes from best prediction to worst
    """

    if not os.path.exists(model_path):
        raise ValueError('No such `{}` file found\n'
                         'Please, checkout the readme of the project '
                         'on github and download required models'.format(model_path))
    model = load_model(model_path)

    # get image as array and resize it if necessary
    pil_img = Image.open(img_path)
    if pil_img.size != img_target_size:
        pil_img = pil_img.resize(img_target_size)

    img = image.img_to_array(pil_img)

    # if alpha channel found, discard it
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # preprocess image
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img).flatten()

    # get predictions index sorted based on the best predictions
    value_ = preds.argsort()
    sorted_preds_index = value_[::-1]

    return preds, sorted_preds_index


def segment_image(img_path):
    """
    Segment leaf from an image and create new segmented image file

    Args:
        img_path: filesystem path of an image

    Returns:
        segmented image file name
    """
    image_name, extension = os.path.splitext(img_path)
    segmented_image_name = image_name + "_marked" + extension  # the future segmented image name to be
    result = subprocess.check_output(['python', "leaf-image-segmentation/segment.py", "-s", img_path])
    print('Info: Input image segmented.')

    return segmented_image_name


def segment_and_predict_species(img_path, model_type=VGG_ARCHITECTURE, do_print=True):
    """
    Given image path, first segment the image and predict species on the segmented image

    Args:
        img_path: filesystem path of an image
        do_print: print information about the prediction
        model_type: type of model to be used for prediction

    Returns:
        a tuple of:
            1. the top one predicted species
            2. segmented image path
    """
    segmented_image_name = segment_image(img_path)

    model_path = os.path.join(MODEL_STORAGE_BASE, get_species_model(model_type))

    target_image_size = TARGET_IMAGE_SIZES[model_type][SPECIES_DETECTION]
    preds, sorted_preds_index = get_predictions(model_path, segmented_image_name, target_image_size)

    if do_print:
        print("Plant Species :")
        for i in sorted_preds_index:
            print("\t - " + str(SPECIES[i]) + ": \t" + str(preds[i]))

    return str(SPECIES[sorted_preds_index[0]]), segmented_image_name


def predict_species(img_path, model_type=VGG_ARCHITECTURE, do_print=True):
    """
    Given an image path, predict the species on the raw image without segmenting

    Args:
        img_path: filesystem path of an image
        do_print: print information about the prediction
        model_type: type of model to be used for prediction

    Returns:
        the top one predicted species
    """

    model_path = os.path.join(MODEL_STORAGE_BASE, get_species_model(model_type))

    target_image_size = TARGET_IMAGE_SIZES[model_type][SPECIES_DETECTION]
    preds, sorted_preds_index = get_predictions(model_path, img_path, target_image_size)

    if do_print:
        print("Plant Species :")
        for i in sorted_preds_index:
            print("\t - " + str(SPECIES[i]) + ": \t" + str(preds[i]))

    return str(SPECIES[sorted_preds_index[0]])


def predict_disease(img_path, species, model_type=VGG_ARCHITECTURE, do_print=True):
    """
    Given an image path and species of the image, predict the disease on the raw image without segmenting

    Args:
        img_path: filesystem path of an image
        species: name of species
        model_type: type of model to be used for prediction
        do_print: print information about the prediction

    Returns:
        the top one predicted disease or None if the species is not supported(has no disease model yet for the species)
    """
    if species not in SPECIES:
        raise ValueError("No such `{}` species is supported.\n"
                         "Supported species are {}".format(species, SPECIES))

    if species not in DISEASE_SUPPORTED_SPECIES:
        print("Info: For `{}` species, a disease can not be predicted "
              "since its disease model is not implemented yet.".format(species))
        return None
    else:
        SPECIES_CLASSES = get_classes(species)
        model_path = os.path.join(MODEL_STORAGE_BASE, get_disease_model(species, model_type))

        target_image_size = TARGET_IMAGE_SIZES[model_type][DISEASE_DETECTION]
        preds, sorted_preds_index = get_predictions(model_path, img_path, target_image_size)

        if do_print:
            print("Plant Disease : ")
            for i in sorted_preds_index:
                print("\t-" + str(SPECIES_CLASSES[i]) + ": \t" + str(preds[i]))

        return str(SPECIES_CLASSES[sorted_preds_index[0]])


def get_cmd_args():
    """
    Get command line arguments if found or use default ones

    Returns:
         list of command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help='Image file path')
    parser.add_argument('--model', type=str.lower, default=VGG_ARCHITECTURE,
                        choices=[VGG_ARCHITECTURE, INCEPTIONV3_ARCHITECTURE],
                        help='Type of model to user for prediction')
    parser.add_argument("--segment", action='store_true', help='Perform segmentation before prediction')
    parser.add_argument("--species", type=str.lower, default='', help='Species Name if known')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_cmd_args()

    # if not segment and species is not known
    if args.segment == False and args.species == '':
        species = predict_species(args.image, args.model)
        predict_disease(args.image, species, args.model)

    # if not segment and species is given
    elif args.segment == False and args.species != '':
        predict_disease(args.image, args.species, args.model)

    # if segment and species is not known
    elif args.segment == True and args.species == '':
        species, image_name = segment_and_predict_species(args.image, args.model)
        predict_disease(image_name, species)

    # if segment and species is given
    elif args.segment == True and args.species != '':
        image_name = segment_image(args.image)
        predict_disease(image_name, args.species, args.model)

    # should not enter here
    else:
        print("Make Sure Your Command is Correct")
