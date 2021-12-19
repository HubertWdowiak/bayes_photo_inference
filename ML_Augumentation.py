
from posix import listdir
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator
from matplotlib.pyplot import cla
import os

from numpy.core.fromnumeric import size




path="/home/data/data/kodeiri/ML_project/dogs datasets/Blind" ##dir to save agumented files
dir_path="/home/data/data/kodeiri/ML_project/dogs datasets/Blind" ## dir to get original photos from
classes=['Adult','Senior','Young'] ## labels
datagen = ImageDataGenerator(channel_shift_range=10,brightness_range=[0.5,1.5],rotation_range=2,horizontal_flip=True) ## augumenting via rotatig the picture by 90 degs
arr=os.listdir(path)
folder_name=arr[0]
#el_num=classes.index(folder_name)
dim = 100

num_of_pics=round(len(os.listdir(dir_path)))



## radnom new line

## for batch size -- set it to the amount of data you want to add to exisitng database
## for save to dir choose an item of classes list that has the same label as the files you are inputting
## to agument data you have to do it folder by folder, starting with adult only etc...

batches=datagen.flow_from_directory(dir_path,class_mode=None,batch_size=num_of_pics,save_to_dir=dir_path,shuffle=True,target_size=(100,100))
print(batches.next()) ## we need that, idk why

