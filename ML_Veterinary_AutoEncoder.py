import tensorflow
from tensorflow import keras
#import keras
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import gzip
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from random import shuffle
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from tensorflow.keras.utils import to_categorical

data=[]
import os
from os import listdir
from os.path import join
import pandas as pd
import numpy as np
import cv2

# from os import listdir
# from os.path import join
# import pandas as pd
# import numpy as np

def extract_data(address,label):
  for filename in listdir(address):
    string_e=address+filename
    img = cv2.imread(string_e)
    IMG_SIZE =28
    resized_image = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # save same img
    cv2.imwrite(string_e, resized_image)

CATEGORIES = ["Adult", "Senior", "Young"]
# we will need to resize all images in files

extract_data('/home/data/data/kodeiri/ML_project/dogs datasets/Experts_train_eval/Adult/','Adult')
extract_data('/home/data/data/kodeiri/ML_project/dogs datasets/Experts_train_eval/Senior/','Senior')
extract_data('/home/data/data/kodeiri/ML_project/dogs datasets/Experts_train_eval/Young/','Young')
extract_data('/home/data/data/kodeiri/ML_project/dogs datasets/Petfinder_All/Adult/','Adult')
extract_data('/home/data/data/kodeiri/ML_project/dogs datasets/Petfinder_All/Senior/','Senior')
extract_data('/home/data/data/kodeiri/ML_project/dogs datasets/Petfinder_All/Young/','Young')
IMG_SIZE=28
training = []
def createTrainingData(path_test,category):
   path = path_test
   class_num = CATEGORIES.index(category)
   for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path,img))
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    training.append([new_array, class_num])

createTrainingData('/home/data/data/kodeiri/ML_project/dogs datasets/Experts_train_eval/Adult/','Adult')
createTrainingData('/home/data/data/kodeiri/ML_project/dogs datasets/Experts_train_eval/Senior/','Senior')
createTrainingData('/home/data/data/kodeiri/ML_project/dogs datasets/Experts_train_eval/Young/','Young')

test = []
def createTestingData(path_test,category):
   path = path_test
   class_num = CATEGORIES.index(category)
   for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path,img))
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    training.append([new_array, class_num])

createTestingData('/home/data/data/kodeiri/ML_project/dogs datasets/Petfinder_All/Adult/','Adult')
createTestingData('/home/data/data/kodeiri/ML_project/dogs datasets/Petfinder_All/Senior/','Senior')
createTestingData('/home/data/data/kodeiri/ML_project/dogs datasets/Petfinder_All/Young/','Young')

shuffle(training)
shuffle(test)

# Assigning Labels and Features
# training data
IMG_SIZE=28
X =[]
y =[]
for features, label in training:
  X.append(features)
  y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Normalising X and converting labels to categorical data
X = X.astype('float32')
X /= 255
from keras.utils import np_utils
# Y = np_utils.to_categorical(y, 4)
Y = np_utils.to_categorical(y, 3)
print(Y[100])
print(np.shape(Y))

# test
# Assigning Labels and Features
#test
IMG_SIZE=28
X_t =[]
y_t =[]
for features, label in test:
  X_t.append(features)
  y_t.append(label)
X_t = np.array(X_t).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Normalising X and converting labels to categorical data
X_t  = X.astype('float32')
X_t  /= 255
from keras.utils import np_utils
# Y = np_utils.to_categorical(y, 4)
Y_t = np_utils.to_categorical(y, 3)
print(Y_t[100])
print(np.shape(Y_t))

# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=X.shape))

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=X_t.shape))

# Create dictionary of target classes
label_dict = {
 0: 'Adult',
 1: 'Senior',
 2: 'Young',
}

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(X[10], (28,28))
curr_lbl = Y[10]
plt.imshow(curr_img, cmap='gray')
#plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(X_t[10], (28,28))
curr_lbl = Y_t[10]
plt.imshow(curr_img, cmap='gray')
#plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# train_data = X.reshape(-1, 28,28, 1)
# # test_data = test_data.reshape(-1, 28,28, 1)
# train_data.shape
# # , test_data.shape
## we already had this

X.dtype
X_t.dtype

# np.max(train_data)
# np.max(test_data)

X = X / np.max(X)
X_t = X_t / np.max(X_t)

print(np.max(X))
print(np.max(X_t))


from sklearn.model_selection import train_test_split
train_X,valid_X,train_ground,valid_ground = train_test_split(X,
                                                             X,
                                                             test_size=0.2,
                                                             random_state=13)

batch_size = 64
epochs = 200
inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))
num_classes = 10

def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):    
    #decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

autoencoder.summary()

autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(200)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

autoencoder.save_weights('autoencoder.h5')
# saving the model

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(Y)
test_Y_one_hot = to_categorical(Y_t)
# not sure about this part because we have it before

# Display the change for category label using one-hot encoding
print('Original label:', Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

train_X,valid_X,train_label,valid_label = train_test_split(X,train_Y_one_hot,test_size=0.2,random_state=13)

print(train_X.shape)
print(valid_X.shape)
print(train_label.shape)
print(valid_label.shape)

def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out

encode = encoder(input_img)
full_model = Model(input_img,fc(encode))

for l1,l2 in zip(full_model.layers[:19],autoencoder.layers[0:19]):
    l1.set_weights(l2.get_weights())

autoencoder.get_weights()[0][1]

full_model.get_weights()[0][1]

for layer in full_model.layers[0:19]:
    layer.trainable = False

full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
full_model.summary()
classify_train = full_model.fit(train_X, train_label, batch_size=64,epochs=100,verbose=1,validation_data=(valid_X, valid_label))
full_model.save_weights('autoencoder_classification.h5')
for layer in full_model.layers[0:19]:
    layer.trainable = True

full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
classify_train = full_model.fit(train_X, train_label, batch_size=64,epochs=100,verbose=1,validation_data=(valid_X, valid_label))
full_model.save_weights('classification_complete.h5')

accuracy = classify_train.history['acc']
val_accuracy = classify_train.history['val_acc']
loss = classify_train.history['loss']
val_loss = classify_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

test_eval = full_model.evaluate(X_t, test_Y_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

predicted_classes = full_model.predict(X_t)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

print(predicted_classes.shape)
print(Y_t.shape)

correct = np.where(predicted_classes==Y_t)[0]
print ("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_t[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], Y_t[correct]))
    plt.tight_layout()

incorrect = np.where(predicted_classes!=Y_t)[0]
print ("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_t[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], Y_t[incorrect]))
    plt.tight_layout()

from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(Y_t, predicted_classes, target_names=target_names))
