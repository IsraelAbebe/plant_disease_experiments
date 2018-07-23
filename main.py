import os
from keras.models import load_model


model_path = os.path.join('Plant_Disease_Detection_Benchmark_models/Models', 'VGG_scratch_segmented_.h5')
model = load_model(model_path)