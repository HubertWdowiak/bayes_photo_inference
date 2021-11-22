import cv2
import matplotlib.pyplot as plt
# utils package is costume made
import sys
import os
from numpy.core.shape_base import block
sys.path.insert(0,'/home/data/data/kodeiri/ML_project')
py_file_location = "/home/data/data/kodeiri/ML_project/utils.py"
sys.path.append(py_file_location)

import utils
from utils import load_class_names
from utils import *

#sys.path.insert(0,'/home/data/kodeiri/ML_project')
import darknet # you need to have it in you local files
from darknet import Darknet


# Set the location and name of the cfg file
# depends were the location of the files are


py_file_location = "/home/data/data/kodeiri/ML_project"
sys.path.append(os.path.abspath('/home/data/data/kodeiri/ML_project/utils.py'))

#from utils import load_class_names
cfg_file = '/home/data/data/kodeiri/ML_project/yolov3.cfg' 

# Set the location and name of the pre-trained weights file
weight_file = '/home/data/data/kodeiri/ML_project/yolov3.weights'

# Set the location and name of the COCO object classes file
namesfile = '/home/data/data/kodeiri/ML_project/coco.names'

# Load the network architecture
m = Darknet(cfg_file)

# Load the pre-trained weights
m.load_weights(weight_file)

# Load the COCO object classes
class_names = load_class_names(namesfile)

# Print the neural network used in YOLOv3
m.print_network()

# probably we don't need this cell
# Set the default figure size
plt.rcParams['figure.figsize'] = [24.0, 14.0]

# Load the image
img = cv2.imread('/home/data/data/kodeiri/ML_project/dogs datasets/Experts_train_eval/Adult/Akita 2-5 F copy.jpg')
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
 
# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
 
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

original_image = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)

# We resize the image to the input width and height of the first layer of the network.    
resized_image = cv2.resize(original_image, (m.width, m.height))

# Display the images
plt.subplot(121)
plt.title('Original Image')
plt.imshow(original_image)
plt.subplot(122)
plt.title('Resized Image')
plt.imshow(resized_image)
plt.show()

# Set the NMS threshold
nms_thresh = 0.6

# Set the IOU threshold
iou_thresh = 0.4

# os.makedirs('/home/data/data/kodeiri/ML_project/train_examples/Senior/HE')
# os.makedirs('/home/data/data/kodeiri/ML_project/train_examples/Young/HE')
# os.makedirs('/home/data/data/kodeiri/ML_project/test_examples/HE')

def old_predict(address): 
# I am not sure is what is sess in professor's code and I am not sure what we need to return from this function
# Set the default figure size
  plt.rcParams['figure.figsize'] = [24.0, 14.0]

# Load the image
# img = cv2.imread('/home/data/data/kodeiri/ML_project/dogs datasets/Experts_train_eval/Adult/Akita 2-5 F copy.jpg')
  img=cv2.imread(address)
  # img = cv2.equalizeHist(img)#histogramic equalization in ahromatic colors
  img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
 
# equalize the histogram of the Y channel
  img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) ## remove hist eq before
 
# convert the YUV image back to RGB format
  img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
# we will use this thing for the function
# Convert the image to RGB
  original_image = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)

# We resize the image to the input width and height of the first layer of the network.    
  resized_image = cv2.resize(original_image, (m.width, m.height))

# Set the IOU threshold. Default value is 0.4
  iou_thresh = 0.4

# Set the NMS threshold. Default value is 0.6
  nms_thresh = 0.6

# Detect objects in the image
  boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
# print(boxes)
# Print the objects found and the confidence level
  print_objects(boxes, class_names)

#Plot the image with bounding boxes and corresponding object class labels
  p_box=plot_boxes(original_image, boxes, class_names, plot_labels = True)

  # return p_box
  return boxes 

# image_address='/home/data/data/kodeiri/ML_project/dogs datasets/Experts_train_eval/Adult/Akita 2-5 F copy.jpg'
# parts=image_address.split("/")
# name=parts[-1]
# # name
# this code here is just for tring to make sure that everything is good with name spliting
import tensorflow as tf

from PIL import Image


# def rescale(im, ratio):
#     size = [int(x*ratio) for x in im.size]
#     im = im.resize(size, Image.ANTIALIAS)
#     return im


#from matplotlib import pyplot as plt
def save_dog_crop(image_address,box):
  image=Image.open(image_address)
  top, left, bottom, right,p,t,o = box
  if torch.is_tensor(top) == True:
    top=top.numpy() 
    top*=100
  # top=np.floor(top+0.5).astype('int32')
    if torch.is_tensor(left) == True:
      left=left.numpy()
      left*=100
  
  # left=np.floor(left+0.5).astype('int32')
    if torch.is_tensor(bottom) == True:
      bottom=bottom.numpy()
      bottom*=1000
  # bottom=np.floor(bottom+0.5).astype('int32')
    if torch.is_tensor(right) == True:
      right=right.numpy()
      right*=1000
  # right=np.floor(right+0.5).astype('int32')

  top=max(0,np.floor(top+0.5).astype('int32'))
  left=max(0,np.floor(left+0.5).astype('int32'))
  bottom=max(image.size[1],np.floor(bottom+0.5).astype('int32'))
  right=max(image.size[1],np.floor(right+0.5).astype('int32'))
  
  box=(left, top, right, bottom)
  print(box)
  # box=(0,0,500,500)
  # box=np.array(box)
  # print(box)

  # crop the pic with this box info
  image= image.crop(box)
  # plt.imshow(image)
  # plt.show()
  # image.show()
  # # # save the cropped image inside of a folder
  parts=image_address.split("/")
  name=parts[-1]
  #path='/home/data/data/kodeiri/ML_project/train_examples/Senior/'+name
  #path='/home/data/data/kodeiri/ML_project/train_examples/Young/'+name
  #path='/home/data/data/kodeiri/ML_project/train_examples/Adult/'+name
  # path='/home/data/data/kodeiri/ML_project/train_examples/Senior/HE/'+name
  # path='/home/data/data/kodeiri/ML_project/train_examples/Adult/HE/'+name
  # path='/home/data/data/kodeiri/ML_project/train_examples/Young/HE/'+name

  path='/home/data/data/kodeiri/ML_project/Petfinder_yolo_prepared/Young/'+name
  #path='/home/data/data/kodeiri/ML_project/train_mix_yolo/Senior/'+name
  #path='/home/data/data/kodeiri/ML_project/train_mix_yolo/Young/'+name

  #the overfitting
  #path='/home/data/data/kodeiri/ML_project/testing_new/Adult/'+name
  #path='/home/data/data/kodeiri/ML_project/testing_new/Senior/'+name
  #path='/home/data/data/kodeiri/ML_project/testing_new/Young/'+name
  newsize=(100,100)
  image=image.resize(newsize)
  rgba = image.convert("RGBA")
  datas = rgba.getdata()
      
  newData = []
  for item in datas:
      if item[0] == 0 and item[1] == 0 and item[2] == 0:  # finding black colour by its RGB value
              # storing a transparent value when we find a black colour
          newData.append((255, 255, 255, 0))
      else:
          newData.append(item)  # other colours remain unchanged
      
  rgba.putdata(newData)
  rgba.save(path, "PNG")  

  #image.save(path)
  # # img = cv2.imread(image)

  # cv2.imwrite ('/home/data/data/kodeiri/ML_project/train_examples'+name, image)

  # # I am suggesting to create another file so there were no need of mixing processed and unprocessed images

  # img = cv2.imread(image_address)
  # crop_img = img[left:left+right, bottom:bottom+top]
  # # plt.imshow(crop_img)
  # cv2.imwrite(os.path.join('/home/data/data/kodeiri/ML_project/train_examples' , name), crop_img)


  # cv2.waitKey(0)

names=[]
classes=[]

from os import listdir
from os.path import join
import pandas as pd
import numpy as np



#need to put back the label argument
def all_dogs_loop(address,label):
  for filename in listdir(address):
    print(address+filename)
    string_e= address+filename
    names.append(string_e)
    classes.append(label)
    box=old_predict(string_e)
    if box == []:
      box= [[0,0,500,500,0,0,0]]
    save_dog_crop(string_e, box[0])


#all_dogs_loop('/home/data/data/kodeiri/ML_project/dogs datasets/Experts_train_eval/Adult/','Adult')
# finished ok after solving the bug

#all_dogs_loop('/home/data/data/kodeiri/ML_project/dogs datasets/Experts_train_eval/Senior/','Senior')

# finished ok after solving the bug

#all_dogs_loop('/home/data/data/kodeiri/ML_project/dogs datasets/Experts_train_eval/Young/','Young')
# finished ok after solving the bug

# petfinder dataset
all_dogs_loop('/home/data/data/kodeiri/ML_project/Petfinder_yolo_prepared/Young/','Senior')
# finished ok 

#all_dogs_loop('/home/data/data/kodeiri/ML_project/dogs datasets/Petfinder_All/Senior/','Senior')
# finished ok after solving the bug

#all_dogs_loop('/home/data/data/kodeiri/ML_project/dogs datasets/Petfinder_All/Young/','Young')
# finished ok after solving the bug

#all_dogs_loop('/home/data/data/kodeiri/ML_project/dogs datasets/n02085620-Chihuahua/')
#it goes through the program without the labels because I couldn't find how they were labeled excatly

#overfitting
#all_dogs_loop('/home/data/data/kodeiri/ML_project/dogs datasets/testing_new/Adult/','Adult')
#all_dogs_loop('/home/data/data/kodeiri/ML_project/dogs datasets/testing_new/Senior/','Senior')
#all_dogs_loop('/home/data/data/kodeiri/ML_project/dogs datasets/testing_new/Young/','Young')


#def datatable_f(address, label):
    #for filename in listdir(address):
    #names.append(filename)
    #classes.append(label)

#datatable_f('/home/data/kodeiri/ML_project/dogs datasets/Experts_train_eval/Adult/','Adult')
#datatable_f('/home/data/kodeiri/ML_project/dogs datasets/Experts_train_eval/Senior/','Senior')
#datatable_f('/home/data/kodeiri/ML_project/dogs datasets/Experts_train_eval/Young/','Young')
#datatable_f('/home/data/kodeiri/ML_project/dogs datasets/Petfinder_All/Young','Young')
#datatable_f('/home/data/kodeiri/ML_project/dogs datasets/Petfinder_All/Senior','Senior')
#datatable_f('/home/data/kodeiri/ML_project/dogs datasets/Petfinder_All/Adult','Adult')

#together=[names, classes]
#df = DataFrame (together,columns=['Names','Classes'])
#df
