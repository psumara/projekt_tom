import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import os
import cv2


def load_image(filename):
    img_nii = nib.load(filename) #load niftii image
    img_numpy = img_nii.get_fdata() #transform niftii format to numpy array
    return img_numpy
    
def generate_center_plot_slices(image):
    #Slice_0 - coronal plane
    #Slice_1 - sagittal plane
    #slice_2 - transverse plane
    slice_0 = image[:, int(image.shape[1] / 2), :]
    slice_1 = image[:, :, int(image.shape[2] / 2)]
    slice_2 = image[int(image.shape[0] / 2), :, :]
    return slice_0, slice_1, slice_2

def generate_axis_view(image):
    #plot image in three planes
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex = True, figsize = (9, 6))
    fig.suptitle("Center slices for NII Image")
    coronal, sagittal, transverse = generate_center_plot_slices(image)
    
    ax1.imshow(transverse.T, cmap = "gray", origin = "lower")
    ax1.set(title = "Transverse plane")
    ax1.axis("off")
    
    ax2.imshow(coronal.T, cmap = "gray", origin = "lower")
    ax2.set(title = "Coronal plane")
    ax2.axis("off")
    
    ax3.imshow(sagittal.T, cmap = "gray", origin = "lower")
    ax3.set(title = "Sagittal plane")
    ax3.axis("off")
    
def get_slices(image, mode = "Transverse"):
    #get image slices in desired plane
    if (mode == "Coronal"):
        image_slices = [image[:, i, :] for i in range(image.shape[1])]
    elif (mode == "Sagittal"):
        image_slices = [image[:, :, i] for i in range(image.shape[2])]
    elif (mode == "Transverse"):
        image_slices = [image[i, :, :] for i in range(image.shape[0])]
    return image_slices


def save_slices(save_path, folder_name, sliced_image, patient_id):
    #save all slices generated from an image  
    for i, image in enumerate(sliced_image):
        cv2.imwrite(os.path.join(str(save_path), str(folder_name) + "_" + str(patient_id) + "_" + str(i) + ".png"), image) 
        
def save_slices_all(path, folder_name):
    #save all slices from a desired folder and generate a new folder to save png files in
    for _, _, files in os.walk(path):
            os.chdir(path)
            new_path = folder_name + "_slices"
            try:
                os.makedirs(new_path)
            except FileExistsError:
                pass
            for i, file in enumerate(files):
                if file.endswith(".nii.gz"):
                    temp = nib.load(file)
                    temp_np = temp.get_fdata()
                    slices = get_slices(temp_np)
                    save_slices(new_path, folder_name, slices, i)        
