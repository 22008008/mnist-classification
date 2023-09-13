# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

MNIST Handwritten Digit Classification Dataset is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.

It is a widely used and deeply understood dataset and, for the most part, is “solved.” Top-performing models are deep learning convolutional neural networks that achieve a classification accuracy of above 99%, with an error rate between 0.4 %and 0.2% on the hold out test dataset.
![image](https://github.com/22008008/mnist-classification/assets/118343520/95bab29d-0a2b-4bd7-bce4-869cfb64dfd9)

## Neural Network Model

![image](https://github.com/22008008/mnist-classification/assets/118343520/95a82416-23d1-4aa2-b8fa-d8ce1f18a912)


## DESIGN STEPS

## STEP 1:
Import tensorflow and preprocessing libraries

## STEP 2: 
Build a CNN model

## STEP 3:

Compile and fit the model and then predict


## PROGRAM

### Importing the required packages:
```
Program developed by : Sri Ranjani Priya P
Ref no : 212222220049

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[0]
single_image.shape

plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0    

X_train_scaled.min()    
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model = keras.Sequential()
model.add (layers. Input (shape=(28,28,1)))
model.add (layers.Conv2D (filters=32, kernel_size=(3,3), activation='relu')) 
model.add (layers.MaxPool2D (pool_size=(2,2)))
model.add (layers. Flatten())
model.add (layers.Dense (32, activation='relu'))
model.add (layers.Dense (10, activation='softmax'))
model.summary()

# Choose the appropriate parameters
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))

## Prediction for a single input
img = image.load_img('/content/PIC-03.png')
type(img)
img = image.load_img('/content/PIC-03.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
     print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single
_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

### Loading the dataset:

(X_train, y_train), (X_test, y_test) = mnist.load_data()

### Shape of training and testing data:

X_train.shape
X_test.shape

### Getting an image at the zeroth index from the training data:

single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')

### Scaling the data:

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

### Implementing one hot encoder:

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

### Scaling training and testing data:

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

### Creating the model:

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1))) 
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu")) 
model.add(layers.MaxPool2D(pool_size=(2,2))) 
model.add(layers.Flatten()) 
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(32)) 
model.add(layers.Dense(10,activation="softmax"))

### Compiling the model:

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

### Fitting the model:

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))

### Creating Dataframe and getting history:

metrics = pd.DataFrame(model.history.history)

### Plotting accuracy vs validated accuracy:

metrics[['accuracy','val_accuracy']].plot()

### Plotting loss vs validated loss:

metrics[['loss','val_loss']].plot()

### Implementing argmax:

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

### Confusion matrix:

print(confusion_matrix(y_test,x_test_predictions))

### Classification report:

print(classification_report(y_test,x_test_predictions))

### Loading an external image:

img = image.load_img('3.jpeg')

### Conversion of the image:

img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

### Prediction:
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/22008008/mnist-classification/assets/118343520/137eab6c-bc88-4b2a-944f-e7391ddd74ce)

![image](https://github.com/22008008/mnist-classification/assets/118343520/b65631ce-7aec-4598-b75c-f5f86cca617a)

### Classification Report

![image](https://github.com/22008008/mnist-classification/assets/118343520/d78b89ad-f542-4b7b-ab23-5bc1b6f0bd15)

### Confusion Matrix

![image](https://github.com/22008008/mnist-classification/assets/118343520/534548ab-1b6d-4bbb-804c-5080348f2942)


### New Sample Data Prediction

![image](https://github.com/22008008/mnist-classification/assets/118343520/284135de-dadd-4f76-b41c-ef37b8541bc9)

## RESULT

Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
