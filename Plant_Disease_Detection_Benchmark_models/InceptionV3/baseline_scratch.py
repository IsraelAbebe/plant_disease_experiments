from __future__ import absolute_import

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

print('tf version: ', tf.__version__)

try:
    from .utils import train_model, nb_classes, INPUT_SHAPE
except:
    from utils import train_model, nb_classes, INPUT_SHAPE


iv3 = InceptionV3(input_shape=INPUT_SHAPE, weights=None,
                  include_top=True, classes=nb_classes)
iv3.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_model(iv3)
