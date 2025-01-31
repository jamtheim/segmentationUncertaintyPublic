# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Class for loading and reading data 
# *********************************************************************************

import numpy as np
import os
import SimpleITK as sitk  
from commonConfig import commonConfigClass
# Load configuration
conf = commonConfigClass() 


class ioDataMethodsClass:
    """
    Class describing functions needed for loading and reading data
    """

    def __init__ (self):
        """
        Init function
        """
        pass


    def readNiftiFile(self, filePath): 
        """
        Read 3D Nifti files to numpy array. 
        Get image resolution in image and the SITK object.
        input: file path for Nifti file
        output: image data in Numpy format, SITK object, image resolution in tuple format
                
        Args:
            filePath (str): Path to the file to be read

        Return:
            None
        """
        assert isinstance(filePath, str), "Input must be string"
        # Read the .nii image containing the volume using SimpleITK.
        # With SimpleITK the image orientation in the numpy array is correct
        sitk_imageData = sitk.ReadImage(filePath)
        # Access the numpy array from the SITK object
        np_imageData = sitk.GetArrayFromImage(sitk_imageData)
        # Get pixelSpacing in image from the SITK object
        pixelSpacing = sitk_imageData.GetSpacing()
        # Input assertion to make sure 3D image
        assert len(np_imageData.shape) == 3, "dim should be 3"
        # Reorder so slices in the 3D stack will be in the last dimension
        np_imageData = np.transpose(np_imageData, (1,2,0))
        # Return np_imagedata, sitk_imageData and pixel spacing
        return np_imageData, sitk_imageData, pixelSpacing

    
    def writeNiftiFile(self, np_imageData, sitk_imageData, outPutFilePath):
        """
        Saves 3D Nifty file

        Args:
            np_imageData (array): 3D numpy array
            sitk_imageData (sitk image): 3D sitk image
            outPutFilePath (str): Path to the file to be saved
 
        Return:
            None
        """
        # Assert numpy array
        assert isinstance(np_imageData, np.ndarray), "Input must be numpy array"
        # Reorder back so slices in the 3D stack will be in the first dimension
        # This is the numpy format from SITK when exported as numpy array
        # Input assertion to make sure 3D image
        assert len(np_imageData.shape) == 3, "dim should be 3"
        np_imageData = np.transpose(np_imageData, (2,0,1))
        # Converting back to SimpleITK 
        # (assumes we didn't move the image in space as we copy the information from the original)
        outImage = sitk.GetImageFromArray(np_imageData)
        outImage.CopyInformation(sitk_imageData)
        # Make sure folder exist before saving 
        os.makedirs(os.path.dirname(outPutFilePath), exist_ok=True)
        # Write the image
        sitk.WriteImage(outImage, outPutFilePath)