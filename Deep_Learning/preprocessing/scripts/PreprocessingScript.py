
# coding: utf-8

# In[1]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some constants 
path = '/home/katya/data/stage1/'


# In[2]:

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


# In[3]:

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


# In[4]:

def resample(image, scan, z=3):
    
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    new_spacing = np.array((z, spacing[1]*2., spacing[2]*2.))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing


# In[5]:

def resize(patients):
    
    pixResizeDict = {}
    zMax = max([z.shape[0] for z in pixResampledDict.values()])
    
    print('Resizing..')
    counter = 0
    
    for patient in patients:
        counter+=1
        z,x,y = pixResampledDict[patient].shape
        if z < zMax:
            extraSlices = np.ones((zMax-z,x,y))*(-1000)
            pixResizeDict[patient] = np.concatenate((pixResampledDict[patient], extraSlices))
            
        else:
            pixResizeDict[patient] = pixResampledDict[patient]
            
        if counter%100 == 0:
            print (counter)
            
    return pixResizeDict


# In[6]:

patients = os.listdir(path)
patients.sort()


# In[7]:

metaList = []

for i in range(0,len(patients),100):
    metaList.append(patients[i:i+100])


# In[8]:

for patients in metaList:
    patientDict = {}
    patientPixelsDict = {}

    print ('Loading scans..')
    
    for patient in patients:
        print (patient)
        patientDict[patient] = load_scan(path + patient)
        patientPixelsDict[patient] = get_pixels_hu(patientDict[patient])
        
    pixResampledDict = {}
    spacingDict = {}

    print ("Resampling..")
    
    for patient in patients:
        
        print (patient)
        pixResampledDict[patient], spacingDict[patient] = resample(patientPixelsDict[patient], patientDict[patient], 3)
        
    del patientPixelsDict
    del patientDict    
    
    pixResizeDict = resize(patients)
    
    for patient in patients:
        np.save('/home/katya/processed_data/'+patient, pixResizeDict[patient])
        
    del pixResampledDict
    del pixResizeDict
    del spacingDict


# In[ ]:



