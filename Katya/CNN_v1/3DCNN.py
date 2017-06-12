
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
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


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

x_train, y_train = data_for_keras(10)
y_train = to_categorical(y_train)
class_weights = {0:len(y_train)/sum(y_train[:,0]), 1:len(y_train)/sum(y_train[:,1])}


# In[4]:

model = Sequential()

model.add(Conv3D(32, (3, 3, 3), activation='relu', data_format='channels_first', input_shape=x_train.shape[1:]))
model.add(Conv3D(32, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2,2,2), data_format='channels_first'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
print(model.summary())

model.fit(x_train, y_train, batch_size=3, nb_epoch=10, class_weight=class_weights)
score = model.evaluate(x_train, y_train, batch_size=1)
print (score)
print (model.predict_proba(x_train))
print ('Done!')

# In[ ]:



