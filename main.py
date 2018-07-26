import os
import numpy as np
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
#import cv2
import subprocess

from keras.applications.inception_v3 import preprocess_input

img_path = 'abc.jpg'
img_seg_path = 'abc_marked.jpg'

target_size = (64,64)

result = subprocess.check_output(['python', "leaf-image-segmentation/generate_marker.py" ,"-s" , 'abc.jpg'])

print(result)

model_path = os.path.join('Plant_Disease_Detection_Benchmark_models/Models', 'VGG_scratch_segmented_dataset.h5')
model = load_model(model_path)

img = Image.open(img_path)
if img.size != target_size:
    img = img.resize(target_size)

x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)
preds = model.predict(x)
print np.argmax(preds)


