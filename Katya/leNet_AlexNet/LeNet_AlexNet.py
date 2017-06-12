
# coding: utf-8

# In[1]:

import numpy as np
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.optimizers import SGD
from keras.utils import to_categorical
import pandas as pd

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


# In[2]:

labelsDF = pd.DataFrame.from_csv('/home/katya/data/stage1_labels.csv', sep=',', index_col=None)
class_weights = {0:len(labelsDF)/sum(labelsDF.cancer), 1:len(labelsDF)/len(labelsDF) - sum(labelsDF.cancer)}
del labelsDF

# Loading patients in the forma
def data_for_keras(number_of_patients):
    
    labelsDF = pd.DataFrame.from_csv('/home/katya/data/stage1_labels.csv', sep=',', index_col=None)

    imgs = []
    labels = []
    tempDF = labelsDF.sample(number_of_patients).reset_index()

    for i in range(len(tempDF)):
        img = np.load('/home/katya/data/processed_data_ResSeg/' + tempDF.id.ix[i] + '.npy')
        img = np.expand_dims(img,axis=0)
        imgs.append(img)
        labels.append(tempDF.cancer.ix[i])

    X = np.array(imgs)

    Y = np.array(labels)
    
    return X, Y


# In[6]:

x_train, y_train = data_for_keras(1)
input_shape = x_train.shape[1:]
del x_train, y_train


# In[7]:

def patientsGenerator():
    for i in range(140):
        x_train, y_train = data_for_keras(10)
        y_train = to_categorical(y_train)
        
        yield x_train, y_train


# In[ ]:

model = Sequential()

model.add(Conv3D(6, (3, 3, 3), activation='tanh', strides=4, data_format='channels_first', input_shape=input_shape))
model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=2, data_format='channels_first'))
model.add(Conv3D(16, (3, 3, 3), strides=1, activation='tanh'))
model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=2, data_format='channels_first'))
model.add(Conv3D(120, (3, 3, 3), strides=1, activation='tanh'))

model.add(Flatten())
model.add(Dense(84, activation='tanh'))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
print(model.summary())

model.fit_generator(patientsGenerator(), steps_per_epoch = 10, nb_epoch = 2, verbose=1, class_weight=class_weights)


# In[25]:

score = model.evaluate(x_train, y_train, batch_size=1)
print (score)
print (model.predict_proba(x_train))


# In[26]:

y_train


# In[ ]:



