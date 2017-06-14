import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
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
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler
from keras.layers.pooling import AveragePooling3D, GlobalMaxPooling3D
from keras.layers import Input, merge, Activation, Dropout
from keras.optimizers import Adamax, Adam, Nadam
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

def convCNNBlock(x, nFilters, dropRate=0.0):
    
    xConv = Conv3D(nFilters, (3,3,3), 
                   border_mode='same', W_regularizer=l2(1e-4), 
                   data_format='channels_first')(x)
    
    xConv = BatchNormalization(axis=1)(xConv)
    
    xConv = GaussianDropout(dropRate)(xConv)
    xConv = LeakyReLU(.1)(xConv)
    
    xConvPool = MaxPooling3D(dim_ordering="th")(xConv)
    
    return xConvPool

def denseCNNBlock(x, name, outSize=1, activation='sigmoid', dropRate=0.0):
    
    xDense = Dense(32, W_regularizer=l2(1e-4))(x)
    xDense = BatchNormalization()(xDense)
    
    xDense = GaussianDropout(dropRate)(xDense)
    xDense = LeakyReLU(.1)(xDense)
    
    xDense = Dense(outSize, activation=activation, 
                   name=name, W_regularizer=l2(1e-3))(xDense)
    
    return xDense

def compileModel(inputShape, dropRate):
    
    x = Input(inputShape)
    
    x1 = convCNNBlock(x, 8, dropRate=dropRate)
    x1Pool = AveragePooling3D(dim_ordering="th")(x)
    x1Merged = merge([x1, x1Pool], mode='concat', concat_axis=1)
    
    x2 = convCNNBlock(x1Merged, 24, dropRate=dropRate)
    x2Pool = AveragePooling3D(dim_ordering="th")(x1Pool)
    x2Merged = merge([x2, x2Pool], mode='concat', concat_axis=1)
    
    x3 = convCNNBlock(x2Merged, 64, dropRate=dropRate)
    x3Pool = AveragePooling3D(dim_ordering="th")(x2Pool)
    x3Merged = merge([x3,x3Pool], mode='concat', concat_axis=1)

    x4 = convCNNBlock(x3Merged, 72, dropRate=dropRate)
    x4Pool = AveragePooling3D(dim_ordering="th")(x3Pool)
    x4Merged = merge([x4, x4Pool], mode='concat', concat_axis=1)

    x5 = convCNNBlock(x4Merged, 72, dropRate=dropRate)
    
    xMaxPool = GlobalMaxPooling3D()(x5)
    xMaxPoolNorm = BatchNormalization()(xMaxPool) 
    
    xMalig = denseCNNBlock(xMaxPoolNorm, name='Malignancy', outSize=1, activation='sigmoid')
    xDiam = denseCNNBlock(xMaxPoolNorm, name='Diameter', outSize=1, activation='sigmoid')
    xLob = denseCNNBlock(xMaxPoolNorm, name='Lobulation', outSize=1, activation='sigmoid')
    xSpic = denseCNNBlock(xMaxPoolNorm, name='Spiculation', outSize=1, activation='sigmoid')
#     xCalc = denseCNNBlock(xMaxPoolNorm, name='Calcification', outSize=1, activation='sigmoid')
#     xSpher = denseCNNBlock(xMaxPoolNorm, name='Sphericity', outSize=1, activation='sigmoid')
    
    model = Model(input=x, output=[xMalig, xDiam, xLob, xSpic])
    
    opt = Nadam(0.01, clipvalue=1.0)
    
    print ('Compiling model...')
    
    model.compile(optimizer=opt,
                  loss={'Malignancy':'mse', 'Diameter':'mse', 'Lobulation':'mse',
                       'Spiculation':'mse'},
                  loss_weights={'Malignancy':5, 'Diameter':1, 'Lobulation':1,
                       'Spiculation':3})
    
    return model


def batchGenerator(xPos, xNeg, ixPos, ixNeg, batchSize, posFraction=0.5):

    df = pd.read_csv('/home/katya/data/CSVFILES/annotations_enhanced.csv')
    
#     #converting strings to arrays:
#     df['calcification'] = df['calcification'].apply(lambda x: strToArr(x, 6))
#     df['sphericity'] = df['sphericity'].apply(lambda x: strToArr(x, 3))
    
#     #for clacification and sphericity creating zero-arrays (to fill up later on):
#     yCalc = np.zeros((df.shape[0],7))
#     ySpher = np.zeros((df.shape[0],4))
    
    #scaling values between 0 and 1:
    yMalig = minmax_scale(df['malignancy'].values)
    yDiam = minmax_scale(df['diameter_mm'].values)
    yLobul = minmax_scale(df['lobulation'].values)
    ySpic = minmax_scale(df['spiculation'].values)
    

    while True:
        
        #calculating the numbers of positive and negative samples to include into a batch
        #according to the positive fraction (posFraction) specified within the parameters:
        pSize = int(posFraction * batchSize)
        nSize = fpSize = int((batchSize - pSize)/2)
        
        pInds = np.random.choice(range(xPos.shape[0]), size=pSize, replace=False)
        nInds = np.random.choice(range(xNeg.shape[0]), size=nSize, replace=False)
#         fpInds = np.random.choice(range(xFP.shape[0]), size=fpSize, replace=False)
        
        xPosBatch, ixPosBatch = xPos[pInds], ixPos[pInds]
        xNegBatch, ixNegBatch = xNeg[nInds], ixNeg[nInds]
#         xFPBatch, ixFPBatch = xFP[fpInds], ixFP[fpInds]
        
        xBatch = np.concatenate([xPosBatch, xNegBatch], axis=0)
        ixBatch = np.concatenate([ixPosBatch, ixNegBatch], axis=0)
        
        #labeling false positive samples (-2) same as all non-nodule samples (-1)
        ixBatch[ixBatch == -2] = -1
        
        #adding a dimension, corresponding to colour channels (we only have 1)
        xBatch = np.expand_dims(xBatch, 1)
        
        #normalizing batch to values between 0 and 1
        xBatch = (xBatch - xBatch.flatten().min()) / (xBatch.flatten().max() - xBatch.flatten().min())
        xBatch = np.clip(xBatch,0,1)
        
        yBatchMalig = yMalig[ixBatch]
        yBatchDiam = yDiam[ixBatch]
        yBatchLobul = yLobul[ixBatch]
        yBatchSpic = ySpic[ixBatch]
        
        #zeroing all parameters of non-nodule samples:
        yBatchMalig[ixBatch == -1] = 0.0
        yBatchDiam[ixBatch == -1] = 0.0
        yBatchLobul[ixBatch == -1] = 0.0
        yBatchSpic[ixBatch == -1] = 0.0
        
        yield xBatch, {'Malignancy':yBatchMalig, 'Diameter':yBatchDiam,
                       'Lobulation':yBatchLobul, 'Spiculation':yBatchSpic}   
        
        
def loadCategory(category):
    
    x = np.load('/home/katya/data/voxels_'+category+'64/subset0X'+category+'.npy')
    ix = np.load('/home/katya/data/voxels_'+category+'64/subset0IX'+category+'.npy')

    for subset in range(1,10):
        xTemp = np.load('/home/katya/data/voxels_'+category+'64/subset' + str(subset) + 'X'+category+'.npy')
        ixTemp = np.load('/home/katya/data/voxels_'+category+'64/subset' + str(subset) + 'IX'+category+'.npy')

        x = np.concatenate((x, xTemp),axis=0)
        ix = np.concatenate((ix, ixTemp),axis=0)
        del xTemp, ixTemp
        
    return x, ix


model = compileModel((1,64,64,64),1e-4)

def train_model_on_stage(model, modelPath, testSize=0.2, batchSize=10, nbEpoch = 1, stepsPerEpoch = 2):
    
    print ('Loading positive patches')
    xPos, ixPos = loadCategory('true')
    xPosTrain,xPosValid,ixPosTrain,ixPosValid = train_test_split(xPos, ixPos, test_size=testSize)   
    del xPos, ixPos
    
    print ('Loading negative patches')
    xNeg, ixNeg = loadCategory('random')
    xNegTrain,xNegValid,ixNegTrain,ixNegValid = train_test_split(xNeg, ixNeg, test_size=testSize)
    del xNeg, ixNeg
    
#     print ('Loading false positive patches')
#     xFP, ixFP = loadCategory('false')
#     xFPTrain,xFPValid,ixFPTrain,ixFPValid = train_test_split(xFP, ixFP, test_size=testSize)
#     del xFP, ixFP

    trainGenerator = batchGenerator(xPosTrain,xNegTrain,
                                    ixPosTrain,ixNegTrain,
                                    batchSize=batchSize,
                                    posFraction=.5)
    
    validGenerator = batchGenerator(xPosValid,xNegValid,
                                    ixPosValid,ixNegValid,
                                    batchSize=batchSize,
                                    posFraction=.5)

    ckp = ModelCheckpoint(filepath=modelPath)
        
#     lossHist = {}
    
#     for lossType in ['loss','val_loss']:
#         lossHist[lossType] = []
    
    for epoch in range(nbEpoch):
        hist = model.fit_generator(trainGenerator, validation_data=validGenerator, 
                                   validation_steps=1,steps_per_epoch=stepsPerEpoch,
                                   nb_epoch=epoch+1,callbacks=[ckp],
                                   initial_epoch=epoch)
        
#         for lossType in ['loss','val_loss']:
#             lossHist[lossType].extend(hist.history[lossType])

    return model, hist

modelPath = '/home/katya/LungCancer/Katya/CNN_v2/model_and_weights/LUNA_model_v2.h5'
model, lossHist = train_model_on_stage(model, modelPath, batchSize=30)

