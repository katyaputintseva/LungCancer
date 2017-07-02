
%matplotlib inline

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some constants 
INPUT_FOLDER = '/home/lin/data/stage1/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
labels = pd.read_csv('/home/lin/data/stage1_labels.csv', index_col=0)s

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[1].ImagePositionPatient[2] - slices[2].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[1].SliceLocation - slices[2].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
    
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

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
#     print (spacing, new_spacing)
    return image, new_spacing


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image



def remove_zeros(nparray):
#     print ('        Original shape..', nparray.shape)
    zaxis = np.stack([z for z in nparray if np.sum(z) != 0])
#     print ('        Shape after z-axis sliced..', zaxis.shape)
    xaxis = np.swapaxes(np.stack([x for x in np.swapaxes(zaxis,0,1) if np.sum(x) != 0]), 0,1)
#     print ('        Shape after x-axis sliced..', xaxis.shape)
    yaxis = np.swapaxes(np.stack([y for y in np.swapaxes(xaxis,0,2) if np.sum(y) != 0]), 0,2)
#     print ('        Shape after y-axis sliced..', yaxis.shape)
    
    return yaxis

def resize(max_dim, seg_lung):
    max_z, max_x, max_y = map(int, max_dim)
    z,x,y = seg_lung.shape
#     print ("Segmented lung dim", seg_lung.shape)
    
    if max_x-x>0:
        zeros_volum = np.ones((z,max_x-x,y))*(-1000)
        seg_lung = np.concatenate((seg_lung, zeros_volum), axis=1)
        z,x,y = seg_lung.shape
        
    if max_y-y>0:
        zeros_volum = np.ones((z,x,max_y-y))*(-1000)
        seg_lung = np.concatenate((seg_lung, zeros_volum), axis=2)
        z,x,y = seg_lung.shape
            
    if max_z-z>0:
        zeros_volum = np.ones((max_z-z,x,y))*(-1000)
        seg_lung = np.concatenate((seg_lung, zeros_volum), axis=0)
        z,x,y = seg_lung.shape
#     print ("Resized lung dim  ", seg_lung.shape)
    return seg_lung

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces, _, _ = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.30)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


