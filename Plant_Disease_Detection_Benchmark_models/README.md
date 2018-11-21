# Plant Disease Detection Benchmark models


**Objective:**
- Using different neural network Archetectures implement Neural Net Prediction Model ,Sothat we may have a baseline model accuracy to compare the future progress

**Resources:**
- [Using Deep Learning for Image-Based Plant Disease Detection](https://arxiv.org/pdf/1604.03169.pdf)
- [Data set](https://github.com/spMohanty/PlantVillage-Dataset)
- [Models](https://gitlab.com/Israel777/Plant_Disease_Detection_models)


## Usage

**AlexNet**

	python alexnet-scratch.py

**InceptionV3**

	python Inception_scratch.py
	
	python finetune.py

**ResNet**
	
	python resnet.py

**VGG**

	python VGG_scratch.py
	
	python finetune.py

				

## Results From the Training (Models that predict from 36 disease classes using PlatVillage Dataset)


**AlexNet**
- Fine tuning  -
- from Scratch - 98%
	    
**InceptionV3**
- fine tuning  - 71%
- from scratch - 94%

**Resnet**
- 97%
	    
**VGG**
- fine tuning  - 80%
- from scratch - 94%



## we also trained Specious calssification networks and specified disease detection models (Using PlantVillage and Some Internal Dataset Combination)

**All Species**
- Based On VGG Arctecture from scratch 94%
- Based On Inception_V3 Arctecture from scratch  99.06%

**Apple**
- Based On VGG Network  93.95%
- Based On Inception_V3 Arctecture from scratch 97.99%

**Cherry**
- Based On VGG Network  98.73%
- Based On Inception_V3 Arctecture from scratch 99.5%

**Corn**
- Based On VGG Network  89.26%
- Based On Inception_V3 Arctecture from scratch 95.83%

**Grape**
- Based On VGG Network  92.93%
- Based On Inception_V3 Arctecture from scratch 99.13%

**Peach**
- Based On VGG Network  97%
- Based On Inception_V3 Arctecture from scratch 98.96%

**Pepper**
- Based On VGG Network  95.90%
- Based On Inception_V3 Arctecture from scratch 99.69%

**Potato**
- Based On VGG Network  90.62%
- Based On Inception_V3 Arctecture from scratch 98.12%

**Strawberry**
- Based On VGG Network  99%
- Based On Inception_V3 Arctecture from scratch 99.5%

**Sugercane**
- Based On VGG Network  83.56%
- Based On Inception_V3 Arctecture from scratch 32.29%

**Tomato**
- Based On VGG Network  86.75%
- Based On Inception_V3 Arctecture from scratch 97.35%









 

