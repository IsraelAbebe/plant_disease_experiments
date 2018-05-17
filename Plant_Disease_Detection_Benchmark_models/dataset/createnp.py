import cv2
import numpy as np
import os         
from random import shuffle 
from tqdm import tqdm      
import tensorflow as tf
import matplotlib.pyplot as plt


TRAIN_DIR = 'color/train'
TEST_DIR = 'color/test'
IMG_SIZE = 100

DIR = 'color/'



classes = []
def create_label():
    """ Create an one-hot encoded vector from image name """
    for i in os.listdir(DIR):
        classes.append(i)


def get_class(image_name):
    # print classes.index(image_name)
    return classes.index(image_name)
    



def create_train_data():
    training_data = []
    for i in tqdm(os.listdir(TRAIN_DIR)):
        for img in os.listdir(os.path.join(TRAIN_DIR, i)):
            path = os.path.join(TRAIN_DIR,i, img)
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            training_data.append([np.array(img_data), get_class(i)])
            
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    print len(training_data)
    return training_data


def create_test_data():
    testing_data = []
    for i in tqdm(os.listdir(TEST_DIR)):
        for img in os.listdir(os.path.join(TEST_DIR, i)):
            path = os.path.join(TEST_DIR,i, img)
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            testing_data.append([np.array(img_data), get_class(i)])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    print len(testing_data)
    return testing_data



def create_data():
    testing_data = []
    for i in tqdm(os.listdir(DIR)):
        for img in os.listdir(os.path.join(DIR, i)):
            path = os.path.join(DIR,i, img)
            img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            testing_data.append([np.array(img_data), get_class(i)])
        
    shuffle(testing_data)
    np.save('data.npy', testing_data)
    print len(testing_data)
    return testing_data



create_label()
create_data()
# create_test_data()
# create_train_data()