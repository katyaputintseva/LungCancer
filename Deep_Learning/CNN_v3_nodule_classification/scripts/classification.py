import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
import os
import pandas as pd
import numpy as np

from keras.models import Model, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers.convolutional import Conv3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout
from keras.optimizers import Adamax, Adam, Nadam
from keras.layers.advanced_activations import ELU,PReLU,LeakyReLU
from keras.activations import relu
from keras.regularizers import l2
from keras.layers.pooling import AveragePooling3D, GlobalMaxPooling3D
from keras.layers import Input, merge, Activation, Dropout
from keras.optimizers import Adamax, Adam, Nadam, sgd
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler,EarlyStopping

from sklearn.preprocessing import minmax_scale
from keras.utils import to_categorical

#######################


def convCNNBlock(x, nFilters, regRate=0.0, dropRate=0.0):
    
    xConv = Conv3D(nFilters, (3,3,3), 
                   border_mode='same', W_regularizer=l2(regRate), 
                   data_format='channels_first')(x)
    
    xConv = BatchNormalization(axis=1)(xConv)
    
    xConv = Dropout(dropRate)(xConv)
    xConv = LeakyReLU()(xConv)
    
    xConvPool = MaxPooling3D(dim_ordering="th")(xConv)
    
    return xConvPool

def denseCNNBlock(x, name, outSize=1, activation='sigmoid', dropRate=0.0, regRate=0.0, neuronNumber=32):
    
    xDense = Dense(neuronNumber, W_regularizer=l2(regRate))(x)
    xDense = BatchNormalization()(xDense)
    
    xDense = Dropout(dropRate)(xDense)
    xDense = LeakyReLU()(xDense)
    
    xDense = Dense(outSize, activation=activation, 
                   name=name, W_regularizer=l2(regRate))(xDense)
    
    return xDense

###############################


def learningRateSchedule(epoch):
    if epoch  < 2:
        return 1e-2
    if epoch < 5:
        return 1e-3
    if epoch < 10:
        return 5e-4
    return 5e-5

def compileModel(inputShape, dropRate, regRate):
    
    x = Input(inputShape)
    
    x1 = convCNNBlock(x, 8, dropRate=dropRate, regRate=regRate)
    x1Pool = AveragePooling3D(dim_ordering="th")(x)
    x1Merged = merge([x1, x1Pool], mode='concat', concat_axis=1)
    
    x2 = convCNNBlock(x1Merged, 24, dropRate=dropRate, regRate=regRate)
    x2Pool = AveragePooling3D(dim_ordering="th")(x1Pool)
    x2Merged = merge([x2, x2Pool], mode='concat', concat_axis=1)
    
    x3 = convCNNBlock(x2Merged, 48, dropRate=dropRate, regRate=regRate)
    x3Pool = AveragePooling3D(dim_ordering="th")(x2Pool)
    x3Merged = merge([x3,x3Pool], mode='concat', concat_axis=1)

    x4 = convCNNBlock(x3Merged, 64, dropRate=dropRate, regRate=regRate)
    x4Pool = AveragePooling3D(dim_ordering="th")(x3Pool)
    x4Merged = merge([x4, x4Pool], mode='concat', concat_axis=1)

    x5 = convCNNBlock(x4Merged, 65, dropRate=dropRate, regRate=regRate)
    
    xMaxPool = GlobalMaxPooling3D()(x5)
    xMaxPoolNorm = BatchNormalization()(xMaxPool) 
    
    xOut = denseCNNBlock(xMaxPoolNorm, name='Nodule', outSize=2, activation='softmax', 
                         dropRate=dropRate, regRate=regRate)
    
    model = Model(input=x, output=xOut)

    opt = sgd(0.01, nesterov=True)
#     opt = Nadam()
    
    print ('Compiling model...')
    
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                metrics=['categorical_accuracy'])
    
    return model

def randomFlips(Xbatch):
    
    swaps = np.random.choice([-1,1],size=(Xbatch.shape[0],3))
    for i in range(Xbatch.shape[0]):
 
        Xbatch[i] = Xbatch[i,::swaps[i,0],::swaps[i,1],::swaps[i,2]]
        
    return Xbatch

def trainModelClass(model, modelPath, testSize=0.2, batchSize=10, nbEpoch = 1, stepsPerEpoch = 2, fp=False):
    
    print ('Loading positive patches')
    xPos = loadCategoryClass('true')
    xPos = randomFlips(xPos)
    xPosTrain,xPosValid,indPosTrain,indPosValid = train_test_split(xPos, 
                                                np.array([n for n in range(xPos.shape[0])]), 
                                                test_size=testSize)
    ixPosTrainClass = np.ones((xPosTrain.shape[0]))
    ixPosValidClass = np.ones((xPosValid.shape[0]))
    del xPos
    
    print ('Loading negative patches')
    xNeg = loadCategoryClass('random')
    
    xNegTrain,xNegValid,indNegTrain,indNegValid = train_test_split(xNeg, 
                                                np.array([n for n in range(xNeg.shape[0])]), 
                                                test_size=testSize)
    ixNegTrainClass = np.zeros((xNegTrain.shape[0]))
    ixNegValidClass = np.zeros((xNegValid.shape[0]))
    del xNeg
    
    trainGenerator = batchGeneratorClass(xPosTrain,xNegTrain,
                                    ixPosTrainClass,ixNegTrainClass,
                                    batchSize=batchSize,
                                    posFraction=.5)

    validGenerator = batchGeneratorClass(xPosValid,xNegValid,
                                    ixPosValidClass,ixNegValidClass,
                                    batchSize=batchSize,
                                    posFraction=.5)
        
    ckp = ModelCheckpoint(filepath=modelPath)
        
    lr = LearningRateScheduler(learningRateSchedule)
    es = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto')
    
    lossHist = {'loss':[], 'val_loss':[], 'val_categorical_accuracy':[], 'categorical_accuracy':[]}
    
    for epoch in range(nbEpoch):
        hist = model.fit_generator(trainGenerator, validation_data=validGenerator, 
                                   validation_steps=10,steps_per_epoch=stepsPerEpoch,
                                   nb_epoch=epoch+1,callbacks=[ckp, lr, es],
                                   initial_epoch=epoch)
        for key in hist.history:
            lossHist[key].extend(hist.history[key])

    return model, lossHist, indPosValid, indNegValid


################################


def batchGeneratorClass(xPos, xNeg, ixPos, ixNeg, batchSize, posFraction=0.5):
    
    while True:
        
        #calculating the numbers of positive and negative samples to include into a batch
        #according to the positive fraction (posFraction) specified within the parameters:
        pSize = int(posFraction * batchSize)
        nSize = fpSize = int((batchSize - pSize)/2)
        
        pInds = np.random.choice(range(xPos.shape[0]), size=pSize, replace=False)
        nInds = np.random.choice(range(xNeg.shape[0]), size=nSize, replace=False)
        
        xPosBatch, ixPosBatch = xPos[pInds], ixPos[pInds]
        xNegBatch, ixNegBatch = xNeg[nInds], ixNeg[nInds]
        
        xBatch = np.concatenate([xPosBatch, xNegBatch], axis=0)
        ixBatch = np.concatenate([ixPosBatch, ixNegBatch], axis=0)
        
        xBatch = np.expand_dims(xBatch, 1)
        
        ixBatch = to_categorical(ixBatch)
        
        yield xBatch, ixBatch 
        
        
def loadCategoryClass(category):
    
    x = np.load('/home/katya/data/voxels_'+category+'64/subset0X'+category+'.npy')

    for subset in range(1,10):
        xTemp = np.load('/home/katya/data/voxels_'+category+'64/subset' + str(subset) + 'X'+category+'.npy')
        x = np.concatenate((x, xTemp),axis=0)
        
        del xTemp
    
    return x


##############################

#Compiling model with branching on the level of Convolution block #5
model = compileModel((1,64,64,64), regRate=1e-3, dropRate=0.2)

# model.load_weights('/home/katya/LungCancer/Katya/CNN_v1/model_v1_weights_temp64.h5')

nbEpochs = 30

modelPath = '/home/katya/LungCancer/Katya/CNN_v3_nodule_classification/model_and_weights/LUNA_model_v3_class.h5'

validInd = {}

model, lossHist, validInd['true'], validInd['random'] = trainModelClass(model, 
                                                            modelPath, 
                                                            batchSize=50, 
                                                            nbEpoch=nbEpochs, 
                                                            stepsPerEpoch=100)

model.save_weights('/home/katya/LungCancer/Katya/CNN_v3_nodule_classification/model_and_weights/LUNA_model_v3_weights_class.h5')


for key in validInd:
    np.save('/home/katya/LungCancer/Katya/CNN_v3_nodule_classification/model_and_weights/valid_inds_'+key+'.npy', validInd[key])
    
version = 1.6

plt.figure(figsize=[10,7])
plt.grid(True, ls='--', lw=0.5, alpha=0.5, dash_capstyle = 'round', c='gray')
plt.xlabel('Epoch #', fontsize=15)
plt.ylabel('Loss', fontsize=15)

plt.plot([x for x in range(7)], lossHist['loss'], 'o-k', label='Training loss')
plt.plot([x for x in range(7)], lossHist['val_loss'], 'o-c', label='Validation loss')
plt.legend()
    
plt.savefig('/home/katya/LungCancer/Katya/CNN_v3_nodule_classification/figures/' + str(version) + '.1.png')


plt.figure(figsize=[10,7])
plt.grid(True, ls='--', lw=0.5, alpha=0.5, dash_capstyle = 'round', c='gray')
plt.xlabel('Epoch #', fontsize=15)
plt.ylabel('Loss', fontsize=15)

plt.plot([x for x in range(nbEpochs)], lossHist['categorical_accuracy'], 'o-k', label='Training categorical accuracy')
plt.plot([x for x in range(nbEpochs)], lossHist['val_categorical_accuracy'], 'o-c', label='Validation categorical accuracy')
plt.legend()
    
plt.savefig('/home/katya/LungCancer/Katya/CNN_v3_nodule_classification/figures/' + str(version) + '.2.png')

