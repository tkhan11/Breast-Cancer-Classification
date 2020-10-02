import os
import cv2
import glob
import numpy as np
import pandas as pd
import time

import sklearn.utils

import matplotlib.pyplot as plt
import random
import fnmatch
import seaborn as sns

from imutils import paths
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import warnings

# IGNORE WARNINGS
warnings.filterwarnings("ignore")

seconds = time.time()
time_start = time.ctime(seconds)   #  The time.ctime() function takes seconds passed as an argument and returns a string representing time.
print("start time:", time_start,"\n") 

images =[]

imagePatches = glob.glob('E:\\ML assignments\\datasets\\BreaKHis_v1\\histology_slides\\breast\\**\\*.png', recursive=True)

'''

#sample imagePaths
for filename in imagePatches[0:30]:
    images.append(filename)
    #print(filename)

# PLOTTING an IMAGE
def plotImage(image_location):
    image = cv2.imread(image_name)
    image = cv2.resize(image,(200,200))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("image",image)
    return

image_name='E:\\ML assignments\\datasets\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB\\adenosis\\SOB_B_A_14-22549AB\\40X\\SOB_B_A-14-22549AB-40-001.png'
#plotImage(image_name)

'''

pattern_benign = '*benign*'
pattern_malignant = '*malignant*'
benign_class = fnmatch.filter(imagePatches, pattern_benign)
malignant_class = fnmatch.filter(imagePatches, pattern_malignant)
print("benign_class\n\n",benign_class[0:2],'\n')
print("malignant_class\n\n",malignant_class[0:2],'\n')


def proc_images(lowerIndex,upperIndex):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """ 
    x = []
    y = []
    WIDTH = 100
    HEIGHT = 100
    for img in imagePatches[lowerIndex:upperIndex]:
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        if img in benign_class:
            y.append(0)          # benign = 0
        elif img in malignant_class:
            y.append(1)       # malignant = 1
        else:
            return
    return x,y

X,Y = proc_images(0,7909)

#dataset=pd.DataFrame({"images":X,"labels":Y})
#dataset.to_csv("new_dataset.csv")

df = pd.DataFrame()
df["images"] = X
df["labels"] = Y
X2 = df["images"]
Y2 = df["labels"]
X2 = np.array(X2)

imgs_benign = []
imgs_malignant = []
imgs_benign = X2[Y2==0]         # benign = 0
imgs_malignant = X2[Y2==1]      # malignant = 1


def describeData(a,b):
    print('Total number of images: {}'.format(len(X2)))
    print('Number of benign Images: {}'.format(np.sum(Y2==0)))
    print('Number of malignant Images: {}'.format(np.sum(Y2==1)))
    print('Image shape (Width, Height, Channels): {}'.format(X2[0].shape))

describeData(X2,Y2)

'''

def plotOne(a,b):
    """
    Plot one numpy array
    """
    plt.subplot(1,2,1)
    plt.title('benign')
    plt.imshow(a[0])
    plt.subplot(1,2,2)
    plt.title('malignant')
    plt.imshow(b[0])
    plt.show()

#plotOne(imgs_benign, imgs_malignant)

#PLOTTING HISTOGRAM
    
def plotHistogram(a):
    """
    Plot histogram of RGB Pixel Intensities
    """
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(a)
    plt.axis('off')
    plt.title('malignant' if Y[1] else 'benign')
    histo = plt.subplot(1,2,2)
    histo.set_ylabel('Count')
    histo.set_xlabel('Pixel Intensity')
    n_bins = 30
    plt.hist(a[:,:,0].flatten(), bins= n_bins, lw = 0, color='r', alpha=0.5);
    plt.hist(a[:,:,1].flatten(), bins= n_bins, lw = 0, color='g', alpha=0.5);
    plt.hist(a[:,:,2].flatten(), bins= n_bins, lw = 0, color='b', alpha=0.5);
    plt.show()

#plotHistogram(X2[100])

'''

X = np.array(X)
X = X/255.0


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# Reduce Sample Size for DeBugging
X_train = X_train[0:300000] 
Y_train = Y_train[0:300000]
X_test = X_test[0:300000] 
Y_test = Y_test[0:300000]

print("\n\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape, '\n')


# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_trainHot = to_categorical(Y_train, num_classes = 2)
Y_testHot = to_categorical(Y_test, num_classes = 2)

'''

#plotting the labels count
lab = df['labels']
dist = lab.value_counts()
sns.countplot(lab)
#plt.show()

'''

# Split the train and the validation set for the fitting
X_train, X_val, Y_trainHot, Y_valHot = train_test_split(X_train, Y_trainHot, test_size = 0.1, random_state=2)

print("\nAfter train and validation set split for model fitting :\n X_train, Y_train, X_validation, Y_validation :",X_train.shape,Y_trainHot.shape,X_val.shape,Y_valHot.shape,'\n')


# Define CNN model architecture 

batch_size = 128
num_classes = 2
epochs = 8  #achieved an accuracy of 85.39% at epochs = 8
img_rows,img_cols=100,100
input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3),
                     activation='relu',
                     input_shape = input_shape,strides=2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images


# Model Training
history = model.fit_generator(datagen.flow(X_train, Y_trainHot, batch_size = 32),
                                  steps_per_epoch = len(X_train) / 32,
                                  epochs = epochs,
                                  validation_data = [X_val, Y_valHot])

print("\n","****************MODEL EVALUATION ************************\n")

# Model Evaluation on Test data
score = model.evaluate(X_test,Y_testHot, verbose=1)
print('\nCNN model #1C - accuracy:', score[1],'\n')

y_pred = model.predict(X_test)
map_characters = {0: 'Benign', 1: 'Malignant'}
              
print('\n',classification_report(np.where(Y_testHot > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')    

Y_pred_classes = np.argmax(y_pred, axis=1) 
Y_true = np.argmax(Y_testHot, axis=1) 

# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

seconds = time.time()
time_stop = time.ctime(seconds)   #  The time.ctime() function takes seconds passed since epoch
print("stop time:", time_stop,"\n")    # as an argument and returns a string representing time.

# saving the model
model.save("breastCancerClassification.h5")


# Training and validation curves

# Plotting the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.savefig("Loss and Accuracy curves.png")
plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()

plot_confusion_matrix(confusion_mtx, classes = range(2))












