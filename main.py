import os
import numpy as np
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
import subprocess
from keras.applications.inception_v3 import preprocess_input
import argparse

parser = argparse.ArgumentParser()

SPECIES = ['Apple','Bean','Blueberry','Cherry','Corn','Grape','Grapefruit','Orange','Peach','Pepper','Potato','Raspberry','Sorghum','Soybean','Squash','Strawberry','Sugarcane','Tomato']
D_MODELS = ['Apple_0.9395_VGG.h5','Cherry_0.9873_VGG.h5','Corn_0.8926_VGG.h5','Grape_0.9293_VGG.h5','Peach_97_VGG.h5','Tomato_0.8675_VGG.h5',
            'pepper_95.90.h5','potato_90.62.h5','starwberry_99.h5','Sugarcane_0.8356_VGG.h5']

APPLE = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy']
CHERRY = ['Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy']
CORN = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy']
GRAPE = ['Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy']
PEACH = ['Peach___Bacterial_spot', 'Peach___healthy']
PEPPER = ['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy']
POTATO = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
STRAWBERRY = ['Strawberry___Leaf_scorch', 'Strawberry___healthy']
SUGERCANE = ['Sugarcane leaf spot', 'Sugarcane aphid', 'Sugarcane coal fouling']
TOMATO = ['Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']



target_size = (64, 64)


def predict(img_path):
    image_name,extension=os.path.splitext(img_path)
    new_image = image_name+"_marked"+extension
    print(new_image)
    result = subprocess.check_output(['python', "leaf-image-segmentation/segment.py", "-s", img_path])
    model_path = os.path.join('Plant_Disease_Detection_Benchmark_models/Models', 'ResNet_0.92.h5')
    model = load_model(model_path)
    img = Image.open(new_image)
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print(np.argmax(preds))


def predict_species(img_path):
    model_path = os.path.join('Plant_Disease_Detection_Benchmark_models/Models', 'VGG_100_0.9497.h5')
    model = load_model(model_path)
    img = Image.open(img_path)
    if img.size != target_size:
        img = img.resize((100,100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x).flatten()
    value_ = preds.argsort()
    value = value_[::-1]
    print("Plant Species ")
    for i in value:
        print("\t - "+str(SPECIES[i])+" : \t"+str(preds[i]))
    return str(SPECIES[value[0]])
    

def predict_disease(img_path,species):
    try:
        CLASS_ARRAY = get_class(species)
        model_path = os.path.join('Plant_Disease_Detection_Benchmark_models/Models', get_model(species))
    except:
        print ('NO Disease Found')
        return 0

    model = load_model(model_path)
    img = Image.open(img_path)
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x).flatten()
    value_ = preds.argsort()
    value = value_[::-1]

    print("Plant Disease : ")
    for i in value:
        print("\t-"+str(CLASS_ARRAY[i])+" : \t"+str(preds[i]))



def get_model(x):
    return {
        'Apple':'Apple_0.9395_VGG.h5',
        'Cherry':'Cherry_0.9873_VGG.h5',
        'Corn':'Corn_0.8926_VGG.h5',
        'Grape':'Grape_0.9293_VGG.h5',
        'Peach':'Peach_97_VGG.h5',
        'Tomato':'Tomato_0.8675_VGG.h5',
        'Pepper':'pepper_95.90.h5',
        'Potato':'potato_90.62.h5',
        'Strawberry':'starwberry_99.h5',
        'Sugarcane':'Sugarcane_0.8356_VGG.h5'
        }[x]


def get_class(x):
    return{
        'Apple' : APPLE,
        'Cherry' : CHERRY,
        'Corn' : CORN,
        'Grape' : GRAPE,
        'Peach' : PEACH,
        'Pepper' : PEPPER,
        'Potato' : POTATO,
        'Strawberry' : STRAWBERRY,
        'Sugarcane' : SUGERCANE,
        'Tomato' : TOMATO,
        }[x]




if __name__ == "__main__":
    parser.add_argument("--image")
    args = parser.parse_args()
    species = predict_species(args.image)
    predict_disease(args.image,species)

