import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import tensorflow as tf

import tensorflow_datasets as tfds
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.optimizers import SGD, RMSprop,Adam
from keras.utils import np_utils
#from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
#from sklearn import metrics
from random import shuffle
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from random import randint
import numpy 
from PIL import Image
import theano

import matplotlib.image as mpimg

import wandb
wandb.login()
from wandb.keras import WandbCallback


def all_img_loop(address,label):
  for filename in listdir(address):
    string_e=address+filename
    img = cv2.imread(string_e)
    IMG_SIZE =100
    resized_image = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # save same img
    cv2.imwrite(string_e, resized_image)

from os import listdir

# we will need to resize all images in files
all_img_loop('/home/data/data/kodeiri/ML_project/train_examples/Adult/','Adult')
all_img_loop('/home/data/data/kodeiri/ML_project/train_examples/Senior/','Senior')
all_img_loop('/home/data/data/kodeiri/ML_project/train_examples/Young/','Young')

IMG_SIZE =100
training = []
path_test = "/home/data/data/kodeiri/ML_project/train_examples/"
CATEGORIES = ["Adult", "Senior", "Young"]

def createTrainingData(path_test,category):
   path = path_test
   class_num = CATEGORIES.index(category)
   for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path,img))
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    training.append([new_array, class_num])

createTrainingData('/home/data/data/kodeiri/ML_project/train_examples/Adult/','Adult')
createTrainingData('/home/data/data/kodeiri/ML_project/train_examples/Senior/','Senior')
createTrainingData('/home/data/data/kodeiri/ML_project/train_examples/Young/','Young')

shuffle(training)

# Assigning Labels and Features
X =[]
y =[]
for features, label in training:
  X.append(features)
  y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# Normalising X and converting labels to categorical data
X = X.astype('float32')
X /= 255
from keras.utils import np_utils
# Y = np_utils.to_categorical(y, 4)
Y = np_utils.to_categorical(y, 3)
print(Y[100])
print(np.shape(Y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
# because the function is returning nympay.ndarray != nympay.array

type(X_train)

# Preparing parametars
batch_size = 16
nb_classes =3
nb_epochs = 25
img_rows, img_columns = 100, 100
img_channel = 3
nb_filters = 32
nb_pool = 2
nb_conv = 3

from keras import Sequential
from tensorflow.keras import layers

# Importing the sigmoid function from
# Keras backend and using it
from keras.backend import sigmoid
  
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

# Getting the Custom object and updating them
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
  
# Below in place of swish you can take any custom key for the name 
get_custom_objects().update({'swish': Activation(swish)})

# Set an experiment name to group training and evaluation
experiment_name = wandb.util.generate_id()

# Start a run, tracking hyperparameters
wandb.init(
  project="dog's age prediction",
  group=experiment_name,
  config={
    "layer_1": 64,
    "activation_1": "leaky_relu",
    "layer_2": 64,
    "activation_2": "leaky_relu",
    "dropout": 0.2,
    "layer_3": 64,
    "activation_3": "leaky_relu",
    "layer_4": 64,
    "activation_4": "leaky_relu",
    "dropout": 0.2,
    "layer_5": 64,
    "activation_5": "leaky_relu",
    "dropout": 0.2,
    "layer_6": 3,
    "activation_6": "softmax",
    "optimizer": "adam",
    "loss": "sparse_categorical_crossentropy",
    "metric": "accuracy",
    "epoch": 25,
    "batch_size": 16
  })
config = wandb.config

#second model
model2=Sequential()

model2.add(layers.Conv2D(64,(7,7),activation='selu',input_shape=(100,100,3), padding="same"))
model2.add(layers.Conv2D(64,(7,7),activation=tf.nn.leaky_relu,padding="same"))

model2.add(layers.BatchNormalization())
model2.add(layers.Dropout(0.2))

model2.add(layers.Conv2D(64,(5,5),activation='selu',padding="same"))
model2.add(layers.Conv2D(64,(5,5),activation=tf.nn.leaky_relu,padding="same"))
model2.add(layers.MaxPool2D(pool_size=(2,2)))

model2.add(layers.BatchNormalization())
model2.add(layers.Dropout(0.2))

model2.add(layers.Flatten())

model2.add(layers.Dense(10, activation='selu'))
model2.add(layers.Dropout(0.2))
model2.add(layers.Dense(3, activation="softmax"))

model2.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model2.fit(X_train, y_train, batch_size = batch_size, epochs = nb_epochs, verbose = 1, validation_data = (X_test, y_test))

score = model2.evaluate(X_test, y_test, verbose = 0 )
print("Test Score: ", score[0])
print("Test accuracy: ", score[1])

y_pred = model2.predict_classes(X_test)

import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False)
#accuracy score
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
print('accuracy of the model: ',score[1])
print (cm)