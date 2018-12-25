import os
import numpy as np
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
import subprocess
from keras.applications.inception_v3 import preprocess_input
import argparse
import sys

SPECIES = ['Apple', 'Bean', 'Blueberry', 'Cherry', 'Corn', 'Grape', 'Grapefruit', 'Orange', 'Peach',
           'Pepper', 'Potato', 'Raspberry', 'Sorghum', 'Soybean', 'Squash', 'Strawberry', 'Sugarcane', 'Tomato']

APPLE = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']
CHERRY = ['Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy']
CORN = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy']
GRAPE = ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
         'Grape___healthy']
PEACH = ['Peach___Bacterial_spot', 'Peach___healthy']
PEPPER = ['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy']
POTATO = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
STRAWBERRY = ['Strawberry___Leaf_scorch', 'Strawberry___healthy']
SUGERCANE = ['Sugarcane leaf spot', 'Sugarcane aphid', 'Sugarcane coal fouling']
TOMATO = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
          'Tomato___healthy']

TARGET_SIZE_DISEASE = (64, 64)
TARGET_SIZE_SPECIES = (100, 100)

# types of models to be used for predictions
VGG_ARCHITECTURE = 'vgg'
INCEPTIONV3_ARCHITECTURE = 'inceptionv3'

# base path from where models will be loaded
MODEL_STORAGE_BASE = 'Plant_Disease_Detection_Benchmark_models/Models'


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
    img = Image.open(img_path)
    if img.size != img_target_size:
        img = img.resize(TARGET_SIZE_SPECIES)
    x = image.img_to_array(img)

    # preprocess image
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x).flatten()

    # get predictions index sorted based on the best predictions
    value_ = preds.argsort()
    sorted_preds_index = value_[::-1]

    return preds, sorted_preds_index


def predict(img_path, do_print=True, model_type=VGG_ARCHITECTURE):
    """
    Given image path, first segment the image and predict species on the segmented image

    Args:
        img_path: filesystem path of an image
        do_print: print information about the prediction
        model_type: type of model to be used for prediction

    Returns:
        a tuple of:
            1. the top one predicted species
            2. read image in PIL image form
    """
    image_name, extension = os.path.splitext(img_path)
    segmented_image_name = image_name + "_marked" + extension
    result = subprocess.check_output(['python', "leaf-image-segmentation/segment.py", "-s", img_path])

    model_path = os.path.join(MODEL_STORAGE_BASE, get_species_model(model_type))

    preds, sorted_preds_index = get_predictions(model_path, segmented_image_name, TARGET_SIZE_SPECIES)

    if do_print:
        print("Plant Species ")
        for i in sorted_preds_index:
            print("\t - " + str(SPECIES[i]) + " : \t" + str(preds[i]))

    return str(SPECIES[sorted_preds_index[0]]), segmented_image_name


def predict_species(img_path, do_print=True, model_type=VGG_ARCHITECTURE):
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

    preds, sorted_preds_index = get_predictions(model_path, img_path, TARGET_SIZE_SPECIES)

    if do_print:
        print("Plant Species ")
        for i in sorted_preds_index:
            print("\t - " + str(SPECIES[i]) + " : \t" + str(preds[i]))

    return str(SPECIES[sorted_preds_index[0]])


def predict_disease(img_path, species, do_print=True, model_type=VGG_ARCHITECTURE):
    """
    Given an image path and species of the image, predict the disease on the raw image without segmenting

    Args:
        img_path: filesystem path of an image
        do_print: print information about the prediction
        model_type: type of model to be used for prediction

    Returns:
        the top one predicted disease
    """
    try:
        CLASS_ARRAY = get_classes(species)
        model_path = os.path.join(MODEL_STORAGE_BASE, get_disease_model(species, model_type))
    except:
        print('NO Disease Found For This Species')
        return 0

    preds, sorted_preds_index = get_predictions(model_path, img_path, TARGET_SIZE_DISEASE)

    if do_print:
        print("Plant Disease : ")
        for i in sorted_preds_index:
            print("\t-" + str(CLASS_ARRAY[i]) + " : \t" + str(preds[i]))

    return str(CLASS_ARRAY[sorted_preds_index[0]])


def get_disease_model(species, model_type):
    """
    Get appropriate disease classifier model file name

    Args:
        species: species name to identify which species model should be used
        model_type: type of model to be used for prediction

    Returns:
        disease classifier model file name
    """

    if model_type == VGG_ARCHITECTURE:
        return {
            'Apple': 'Apple_0.9395_VGG.h5',
            'Cherry': 'Cherry_0.9873_VGG.h5',
            'Corn': 'Corn_0.8926_VGG.h5',
            'Grape': 'Grape_0.9293_VGG.h5',
            'Peach': 'Peach_97_VGG.h5',
            'Tomato': 'Tomato_0.8675_VGG.h5',
            'Pepper': 'pepper_95.90.h5',
            'Potato': 'potato_90.62.h5',
            'Strawberry': 'starwberry_99.h5',
            'Sugarcane': 'Sugarcane_0.8356_VGG.h5'
        }[species]
    elif model_type == INCEPTIONV3_ARCHITECTURE:
        return {
            'Apple': 'InceptionV3-scratch_segApple.h5 ',
            'Cherry': 'InceptionV3-scratch_segCherry.h5',
            'Corn': 'InceptionV3-scratch_segCorn.h5 ',
            'Grape': 'InceptionV3-scratch_segGrape.h5 ',
            'Peach': 'InceptionV3-scratch_segPeach.h5 ',
            'Tomato': 'InceptionV3-scratch_segTomato.h5 ',
            'Pepper': 'InceptionV3-scratch_segPepper.h5 ',
            'Potato': 'InceptionV3-scratch_segPotato.h5 ',
            'Strawberry': 'InceptionV3-scratch_segStrawberry.h5 ',
            'Sugarcane': 'InceptionV3-scratch_segSugarcane.h5 '
        }[species]


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


def get_classes(species_name):
    """
    Get classes of disease for a species

    Args:
        species_name: name of species

    Returns:
        a list of disease classes for a specific species
    """
    return {
        'Apple': APPLE,
        'Cherry': CHERRY,
        'Corn': CORN,
        'Grape': GRAPE,
        'Peach': PEACH,
        'Pepper': PEPPER,
        'Potato': POTATO,
        'Strawberry': STRAWBERRY,
        'Sugarcane': SUGERCANE,
        'Tomato': TOMATO,
    }[species_name]


if __name__ == "__main__":
    # get cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help='image path')
    parser.add_argument("--segment", type=bool, default=False, help='add segmentation')
    parser.add_argument("--species", type=str, default='', help='Species Name if known')
    parser.add_argument('--model_type', default=VGG_ARCHITECTURE, choices=[VGG_ARCHITECTURE, INCEPTIONV3_ARCHITECTURE],
                        help='Type of model')
    args = parser.parse_args()

    # if not segment and species is not known
    if args.segment == False and args.species == '':
        species = predict_species(args.image, args.model_type)
        predict_disease(args.image, species, args.model_type)

    # if segment and species is not known     
    elif args.segment == True and args.species == '':
        species, image_name = predict(args.image, args.model_type)
        predict_disease(image_name, species)

    # if segment and species is given
    elif args.segment == True and args.species != '':
        species, image_name = predict(args.image, False, args.model_type)
        predict_disease(image_name, species, args.model_type)

    # if not segment and species is given
    elif args.segment == False and args.species != '':
        predict_disease(args.image, args.species, args.model_type)

    # should not enter here
    else:
        print("Make Sure Your Command is Correct")
