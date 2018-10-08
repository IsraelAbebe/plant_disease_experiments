import os
import numpy as np
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
import subprocess
from keras.applications.inception_v3 import preprocess_input
import argparse

parser = argparse.ArgumentParser()

# img_path = 'abc.jpg'
# img_seg_path = 'abc_marked.jpg'

target_size = (64, 64)


def predict(img_path):
    image_name,extension=os.path.splitext(img_path)
    new_image = image_name+"_marked"+extension
    print(new_image)
    result = subprocess.check_output(['python', "leaf-image-segmentation/segment.py", "-s", img_path])
    model_path = os.path.join('Plant_Disease_Detection_Benchmark_models/Models', 'VGG_scratch_segmented_.h5')
    model = load_model(model_path)
    img = Image.open(new_image)
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print(np.argmax(preds))


if __name__ == "__main__":
    parser.add_argument("--image")
    args = parser.parse_args()
    predict(args.image)

