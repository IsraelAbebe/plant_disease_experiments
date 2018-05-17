# Plant Disease Detection GAN experimants - on [PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)


Experementation on different Gnerative Networks to be able to generate plant image dataset that can be used to train the model and increase the accuracy.


**Objective**

A) to learn a generative network on the whole database.
    
B) to learn a model on the set of plants possessing a particular disease.
    
C) to learn a model on the set of plants belonging to a particular species.



**Dataset**

Use raw folder from [PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset) in each folder
    
    
    
## Usage

**vanilla gan**

    # in vanilla-gan.py give the directory of your dataset folder
    plant_data = Dataset('raw/grayscale/') 
    
    # to run
    python vanilla-gan.py
    
  
**dc-gan-color**

    python gan.py
   
**dc-gan**
     
     #in dc-gan.py
     plant_data = Dataset('raw/grayscale/')
     
    #to run 
    python dc-gan.py
    
**info-wgan**

    # in info-wgan.py
    plant_data = Dataset('raw/grayscale/')
    
    #to run
    python info-wgan.py
    
**infoDCGAN**

    # in infoDCGAN.py
    plant_data = Dataset('raw/grayscale/')
    
    #to run
    python infoDCGAN.py
  

    

