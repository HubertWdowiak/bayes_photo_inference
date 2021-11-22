
from posix import listdir
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator
from matplotlib.pyplot import cla
import os

from numpy.core.fromnumeric import size


##random test comment 
## still commentign to get used to git


path="/home/data/data/kodeiri/ML_project/train_aug_buffer/" ##dir to save agumented files
dir_path="/home/data/data/kodeiri/ML_project/train_aug_buffer" ## dir to get original photos from
classes=['Adult','Senior','Young'] ## labels
datagen = ImageDataGenerator(channel_shift_range=10,brightness_range=[0.5,1.5],rotation_range=2,horizontal_flip=True) ## augumenting via rotatig the picture by 90 degs
arr=os.listdir(path)
folder_name=arr[0]
el_num=classes.index(folder_name)
dim = 100

num_of_pics=round(len(os.listdir(path+classes[el_num]))*0.47)



## radnom new line

## for batch size -- set it to the amount of data you want to add to exisitng database
## for save to dir choose an item of classes list that has the same label as the files you are inputting
## to agument data you have to do it folder by folder, starting with adult only etc...

batches=datagen.flow_from_directory(dir_path,class_mode='categorical',batch_size=num_of_pics,classes=classes,save_to_dir=(path+classes[el_num]),shuffle=True,target_size=(100,100))
print(batches.next()) ## we need that, idk why

