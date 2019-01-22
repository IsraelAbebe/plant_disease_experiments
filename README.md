# Plant Disease Detection using Deep Learning

[Using Deep Learning for Image-Based Plant Disease Detection](https://arxiv.org/pdf/1604.03169.pdf) 



**Resources:**
- [Data set](https://github.com/spMohanty/PlantVillage-Dataset)

- Additional dataset will be relesed 

- [Models](https://gitlab.com/Israel777/Plant_Disease_Detection_models)



**Objective**

- Train and Evaluate different DNN Models for plant deasise detection Problem

- To tackle the problem of scarce real-life representative data, experiment with different generative networks and generate more plant leaf image data

- Implement segmentation pipeline to avoid missclassification due to unwanted input 



**Approches for Solving the papers realtime Detection Problem**

phase 1 : [implement the paper](https://github.com/singnet/plant-disease-experiments/tree/master/Plant_Disease_Detection_Benchmark_models) 

phase 2 : do analysis on the paper and identify the type of data problem 

phase 3 : [experement and if possible generate Apprprate data
		  using the data train the model again](https://github.com/singnet/plant_disease_experements/tree/master/Plant_Disease_Detection_gan_experimants)


# Project Structure

**Plant_Disease_Detection_Benchmark_models**

- Train and test different prediction models to get a baseline accuracy to compare to and see progress

**Plant_Disease_Detection_gan_experiments**

- experiment with different generative networks to see their generative capability and if the output can be used to train more robust models

**leaf-image-segmentation-segnet**

- segmentation pipline using VGGSegNet Architecture

**leaf-image-segmentation**

- histogram based segmentation Pipline 





# Usage

	Python main.py IMAGE_FILE [--segment BOOLEAN_PARAMETER] [--species SPECIES_TYPE] [--model PREDICTION_MODEL]

	arguments

		- IMAGE_FILE    loaction of the image
		- --segment     True to  Segment before prediction , False not to 
		- --species     one of the Following Specious :  Apple,Cherry,Corn, Grape,Peach, Pepper,Potato,Strawberry, Sugercane, Tomato
		- --model       what models do you want to use, vgg or inceptionv3



# Examples


		 	# you can remove a part of arguments except image path

		 >>  python main.py "test/a.jpg" --segment True --species "Apple" --model 'inceptionv3'
	   

- before using that make sure you download the weights from   [here for Inception_V3](https://drive.google.com/file/d/1PZ0SUyGbcKJidNcSfwKsnhR23O2PBl78/view?usp=sharing) and  [here for VGG Models](https://drive.google.com/file/d/1AufdWYl-TfeicAmaweq6Gd8q3--vuBfA/view?usp=sharing)  and extract all and put it in Plant_Disease_Detection_Benchmark_models/Models/  folder 
		
- This will segment the image and predict the output class based on that . segmented image will be saved as the file name with "_masked" prefix.


- the images are trained with segmented network and lower performance on unsegmented dataset is expected 

-  You can cheack the segmentation accuracy from saved image


- Fill [this form](https://goo.gl/forms/ceQNkEimLL8NN1sF2) for bulk model accsess grants and future update Notification
