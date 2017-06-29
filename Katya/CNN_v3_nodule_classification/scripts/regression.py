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

######################


def learningRateSchedule(epoch):
    if epoch  < 2:
        return 1e-2
    if epoch < 5:
        return 1e-3
    if epoch < 10:
        return 5e-4
    return 5e-5

def compileModelDeepBranching(inputShape, regRate, dropRate):
    
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
    
    
    ###Malignancy###
    x5Malig = convCNNBlock(x4Merged, 65, dropRate=dropRate, regRate=regRate)
    xMaxPoolMalig = GlobalMaxPooling3D()(x5Malig)
    xMaxPoolNormMalig = BatchNormalization()(xMaxPoolMalig) 
    xMalig = denseCNNBlock(xMaxPoolNormMalig, name='Malignancy', 
                           outSize=1, activation='softplus', 
                           dropRate=dropRate, regRate=regRate)
    
    ###Diameter###
    x5Diam = convCNNBlock(x4Merged, 65, dropRate=dropRate, regRate=regRate)
    xMaxPoolDiam = GlobalMaxPooling3D()(x5Diam)
    xMaxPoolNormDiam = BatchNormalization()(xMaxPoolDiam) 
    xDiam = denseCNNBlock(xMaxPoolNormDiam, name='Diameter', 
                          outSize=1, activation='softplus', 
                          dropRate=dropRate, regRate=regRate)
    
    ###Lobulation###
    x5Lob = convCNNBlock(x4Merged, 65, dropRate=dropRate, regRate=regRate)
    xMaxPoolLob = GlobalMaxPooling3D()(x5Lob)
    xMaxPoolNormLob = BatchNormalization()(xMaxPoolLob) 
    xLob = denseCNNBlock(xMaxPoolNormLob, name='Lobulation', 
                         outSize=1, activation='softplus', 
                         dropRate=dropRate, regRate=regRate)
    
    ###Spiculation###
    x5Spic = convCNNBlock(x4Merged, 65, dropRate=dropRate, regRate=regRate)
    xMaxPoolSpic = GlobalMaxPooling3D()(x5Spic)
    xMaxPoolNormSpic = BatchNormalization()(xMaxPoolSpic) 
    xSpic = denseCNNBlock(xMaxPoolNormSpic, name='Spiculation', 
                          outSize=1, activation='softplus', 
                          dropRate=dropRate, regRate=regRate)

    
    model = Model(input=x, output=[xMalig, xDiam, xLob, xSpic])

#     opt = Nadam(0.01, clipvalue=1.0)
    opt = sgd(0.01, nesterov=True)
    
    print ('Compiling model...')
    
    model.compile(optimizer=opt,
                  loss={'Malignancy':'mse', 'Diameter':'mse', 'Lobulation':'mse',
                       'Spiculation':'mse'},
                  loss_weights={'Malignancy':1, 'Diameter':1, 'Lobulation':1,
                       'Spiculation':1})
    
    return model


def randomFlips(Xbatch):
    
    swaps = np.random.choice([-1,1],size=(Xbatch.shape[0],3))
    for i in range(Xbatch.shape[0]):
 
        Xbatch[i] = Xbatch[i,::swaps[i,0],::swaps[i,1],::swaps[i,2]]
        
    return Xbatch

def trainRegressionModel(model, modelPath, modelPathClass, 
                         validInd, testSize=0.2, batchSize=10, 
                         nbEpoch=1, stepsPerEpoch=2, fp=False,
                         posFraction=0.5):
    
#     print ('Loading positive patches')
#     xPosTrain, ixPosTrain, xPosValid, ixPosValid = nodulePredictor('true', modelPathClass, validInd)

#     print ('Loading negative patches')
#     xNegTrain, ixNegTrain, xNegValid, ixNegValid = nodulePredictor('random', modelPathClass, validInd)
    
#     print ('Loading false positive patches')
#     xFP, ixFP = loadCategory('false')
#     xFPTrain,xFPValid,ixFPTrain,ixFPValid = train_test_split(xFP, ixFP, test_size=testSize)
#     del xFP, ixFP

    if fp==False:
        trainGenerator = batchGeneratorRegression(xPosTrain,xNegTrain,
                                        ixPosTrain,ixNegTrain,
                                        batchSize=batchSize, 
                                        posFraction=posFraction, fp=False)
        
    else:
        trainGenerator = batchGeneratorRegression(xPosTrain, xNegTrain,
                                ixPosTrain, ixNegTrain,
                                batchSize=batchSize,
                                xFP=xFP, ixFP=ixFP, 
                                posFraction=posFraction, fp=True)

    validGenerator = batchGeneratorRegression(xPosValid,xNegValid,
                                    ixPosValid,ixNegValid,
                                    batchSize=batchSize,
                                    posFraction=posFraction)
        
        
    ckp = ModelCheckpoint(filepath=modelPath)
        
    lossHist = {}
    
    for lossType in ['Diameter_loss', 'val_loss', 
                     'val_Lobulation_loss', 'Spiculation_loss', 
                     'loss', 'val_Diameter_loss', 'val_Spiculation_loss', 
                     'Lobulation_loss', 'val_Malignancy_loss', 'Malignancy_loss']:
        
        lossHist[lossType] = []
        
    lr = LearningRateScheduler(learningRateSchedule)
    es = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=4, verbose=0, mode='auto')
    
    for epoch in range(nbEpoch):
        
        hist = model.fit_generator(trainGenerator, validation_data=validGenerator, 
                                   validation_steps=20,steps_per_epoch=stepsPerEpoch,
                                   nb_epoch=epoch+1,callbacks=[ckp, lr, es],
                                   initial_epoch=epoch)
        
        for lossType in hist.history.keys():
            lossHist[lossType].extend(hist.history[lossType])

    return model, lossHist


################################


def batchGeneratorRegression(xPos, xNeg, ixPos, ixNeg, 
                             batchSize, xFP=None, ixFP=None, 
                             fp=False, posFraction=0.5):

    df = pd.read_csv('/home/katya/data/CSVFILES/annotations_enhanced.csv')

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
        
        xPosBatch, ixPosBatch = xPos[pInds], ixPos[pInds]
        xNegBatch, ixNegBatch = xNeg[nInds], ixNeg[nInds]
        
        if fp==True:
            
            fpInds = np.random.choice(range(xFP.shape[0]), size=fpSize, replace=False)
            xFPBatch, ixFPBatch = xFP[fpInds], ixFP[fpInds]
            
            xBatch = np.concatenate([xPosBatch, xNegBatch, xFPBatch], axis=0)
            ixBatch = np.concatenate([ixPosBatch, ixNegBatch, ixFPBatch], axis=0)            

        else:
            
            xBatch = np.concatenate([xPosBatch, xNegBatch], axis=0)
            ixBatch = np.concatenate([ixPosBatch, ixNegBatch], axis=0)
        
        #labeling false positive samples (-2) same as all non-nodule samples (-1)
        ixBatch[ixBatch == -2] = -1
        
        #adding a dimension, corresponding to colour channels (we only have 1)
#         xBatch = np.expand_dims(xBatch, 1)
        
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
        
    print ('The number of samples for %s category constitutes %d' % (category, len(ix)))
    
    return x, ix



def nodulePredictor(category, modelPath, validInd):

    model = load_model(modelPath)
    x, ix = loadCategory(category)
    
    x = np.expand_dims(x, 1)
    
    xVal = x[validInd[category]]
    ixVal = ix[validInd[category]]
    
    xTrain = np.array([n for i,n in enumerate(x) if i not in validInd[category]])
    ixTrain = np.array([n for i,n in enumerate(ix) if i not in validInd[category]])
    
    print ('Predicting...')
    
    yValidHat = model.predict(xVal, batch_size=10, verbose=1)
    yTrainHat = model.predict(xTrain, batch_size=10, verbose=1)
    
    posValInds = np.where(yValidHat[:,1]>0.5)
    posTrainInds = np.where(yTrainHat[:,1]>0.5)
    
    nodulesVal = xVal[posValInds]
    iNodulesVal = ixVal[posValInds]
    
    print ('Number of predicted validation nodules is %d' % iNodulesVal.shape[0])
    
    nodulesTrain = xTrain[posTrainInds]
    iNodulesTrain = ixTrain[posTrainInds]  
    
    print ('Number of predicted train nodules is %d' % iNodulesTrain.shape[0])
    print ('-------------------------------------------------------------------')
    
    return nodulesTrain, iNodulesTrain, nodulesVal, iNodulesVal


###################################


features = ['Malignancy', 'Diameter', 'Lobulation', 'Spiculation']

validInd = {}
for key in ['random', 'true']:
    validInd[key] = np.load('/home/katya/LungCancer/Katya/CNN_v3_nodule_classification/model_and_weights/valid_inds_'+key+'.npy')
    
    
modelPath = '/home/katya/LungCancer/Katya/CNN_v3_nodule_classification/model_and_weights/LUNA_model_v3_regression.h5'
modelPathClass = '/home/katya/LungCancer/Katya/CNN_v3_nodule_classification/model_and_weights/LUNA_model_v3_class.h5'


print ('Loading positive patches')
xPosTrain, ixPosTrain, xPosValid, ixPosValid = nodulePredictor('true', modelPathClass, validInd)

print ('Loading negative patches')
xNegTrain, ixNegTrain, xNegValid, ixNegValid = nodulePredictor('random', modelPathClass, validInd)

print ('Loading false positive patches')
xFP, ixFP = loadCategory('false')


#Compiling model with branching on the level of Convolution block #5
model = compileModelDeepBranching((1,64,64,64), dropRate=0.3, regRate=1e-3)

nbEpochs = 30

model, lossHist = trainRegressionModel(model, modelPath=modelPath, modelPathClass=modelPathClass,
                                       validInd=validInd, posFraction=0.7,
                                       batchSize=50, nbEpoch=nbEpochs, stepsPerEpoch=100)

model.save_weights('/home/katya/LungCancer/Katya/CNN_v3_nodule_classification/model_and_weights/LUNA_model_v3_weights_regression.h5')

version = 2.4

plt.figure(figsize=[10,7])
plt.grid(True, ls='--', lw=0.5, alpha=0.5, dash_capstyle = 'round', c='gray')
plt.xlabel('Epoch #', fontsize=15)
plt.ylabel('Loss', fontsize=15)

labels=list(lossHist.keys())

for i,key in enumerate(lossHist):
    plt.plot([x for x in range(nbEpochs)], lossHist[key], 'o-', label=labels[i])
    plt.legend()
    
plt.savefig('/home/katya/LungCancer/Katya/CNN_v3_nodule_classification/figures/' + str(version) + '.png')

for feature in features:
    plt.figure(figsize=[10,8])
    plt.grid(True, ls='--', lw=0.5, alpha=0.5, dash_capstyle = 'round', c='gray')

    plt.plot([x for x in range(nbEpochs)], lossHist[feature+'_loss'], label=feature+'_loss')
    plt.plot([x for x in range(nbEpochs)], lossHist['val_'+feature+'_loss'], label='val_'+feature+'_loss')
    plt.title(feature)
    plt.legend()
    plt.savefig('/home/katya/LungCancer/Katya/CNN_v3_nodule_classification/figures/'+str(version)+'.'+feature+'.png')