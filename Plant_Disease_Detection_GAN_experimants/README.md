# Plant Disease Detection using Deep Learning

<<<<<<< HEAD:Plant_Disease_Detection_GAN_experimants/README.md
<<<<<<< HEAD
[Using Deep Learning for Image-Based Plant
Disease Detection](https://arxiv.org/pdf/1604.03169.pdf)
=======


***Objective***
>>>>>>> f3663ee... reverting
=======
[Using Deep Learning for Image-Based Plant Disease Detection](https://arxiv.org/pdf/1604.03169.pdf) 

**Resources:**
- [Data set](https://github.com/spMohanty/PlantVillage-Dataset)

- [Models](https://gitlab.com/Israel777/Plant_Disease_Detection_models)


>>>>>>> 3deaa1e... Update README.md:README.md

**Objective**

- Train and Evaluate different DNN Models for plant deasise detection Problem

- To tackle the problem of scarce real-life representative data, experiment with different generative networks and generate more plant leaf image data

- Implement segmentation pipeline to avoid missclassification due to unwanted input 


<<<<<<< HEAD:Plant_Disease_Detection_GAN_experimants/README.md
<<<<<<< HEAD
=======

**Approches for Solving the papers realtime Detection Problem**
>>>>>>> 3deaa1e... Update README.md:README.md

phase 1 : [implement the paper](https://github.com/singnet/plant-disease-experiments/tree/master/Plant_Disease_Detection_Benchmark_models) 

phase 2 : do analysis on the paper and identify the type of data problem 

phase 3 : [experement and if possible generate Apprprate data
		  using the data train the model again](https://github.com/singnet/plant_disease_experements/tree/master/Plant_Disease_Detection_gan_experimants)

<<<<<<< HEAD:Plant_Disease_Detection_GAN_experimants/README.md
=======
***Dataset***
User raw folder from [PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset) in each folder
>>>>>>> f3663ee... reverting
=======

# Project Structure

**Plant_Disease_Detection_Benchmark_models**

- Train and test different prediction models to get a baseline accuracy to compare to and see progress

**Plant_Disease_Detection_gan_experiments**

- experiment with different generative networks to see their generative capability and if the output can be used to train more robust models

**leaf-image-segmentation-segnet**

- segmentation pipline using VGGSegNet Architecture

<<<<<<< HEAD:Plant_Disease_Detection_GAN_experimants/README.md
>>>>>>> 3deaa1e... Update README.md:README.md
=======
**leaf-image-segmentation**

- histogram based segmentation Pipline 

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> c4886c9... Update README.md
=======
>>>>>>> 4bcacea... Update README.md
=======




# Usage

	Python main.py [--image  IMAGE FILE] [--segment BOOLIAN PARAMETER] [--species SPECIES TYPE]

	arguments

		- --image        a path of image 
		- --segment     True to  Segment before prediction , False not to 
		- --species     one of the Following Specious :  Apple,Cherry,Corn, Grape,Peach, Pepper,Potato,Strawberry, Sugercane, Tomato


- before using that make sure you download the weights from [here](https://drive.google.com/file/d/1AufdWYl-TfeicAmaweq6Gd8q3--vuBfA/view?usp=sharing) and put it in Plant_Disease_Detection_Benchmark_models/Models/  folder 
		
- This will segment the image and predict the output class based on that . segmented image will be saved as the file name with "_masked" prefix.


- the images are traine with segmented network and lower performance on unsegmented dataset is expected 

-  You can cheack the segmentation accuracy from saved image
>>>>>>> 43ed67a... Readme Update
>>>>>>> da5852f... histogram based segmentation module by Yared Taddese:README.md
