# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Class for evaluation of observer data with different methods and metrics
# *********************************************************************************

import numpy as np
import os
import random 
import pydicom
import nibabel as nib
from scipy.ndimage import center_of_mass
import scipy.ndimage as sndim
from platipy.imaging.label.comparison import (
    compute_metric_dsc,
    compute_volume,
    compute_metric_hd,
    compute_metric_masd,
    compute_volume_metrics,
    compute_surface_metrics,
    compute_surface_dsc,
    compute_apl,
    compute_metric_mean_apl,
    compute_metric_total_apl,
    compute_metric_sensitivity,
    compute_metric_specificity,
    )
import surface_distance
import SimpleITK as sitk  
from commonConfig import commonConfigClass
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
import matplotlib.pyplot as plt
import csv

# Load configuration
conf = commonConfigClass() 


class evalDataMethodsClass:
    """
    Class describing functions needed for evaluation of observer data
    """

    def __init__ (self):
        """
        Init function
        """
        pass


    def surfaceDistancesSigned(self, structure1, structure2, voxelSizes):
        """
        Returns the signed surface distances between structure1 and structure2.
        Original code provided by Federica Carmen Maruccio, M.Sc., Biomedical Engineer,
        Department of Radiotherapy, NKI-AVL, Amsterdam, The Netherlands.
        Adapted by Christian Jamtheim Gustafsson, PhD, Medical Physicist Expert, 
        kind of just adjusting variable names.
        Function in Deep Mind package only return a sorted list of distances.
        Therefore this function is needed to get the signed distances when calculating inter-observer differences. 
        Args: 
            structure1: np.array of the first structure
            structure2: np.array of the second structure
            voxel_sizes: np.array containing the voxel size in the 3 dimensions
        Returns:
            np.array: Signed surface distances between structure1 and structure2
        """
        # Assert inputs
        assert structure1.shape == structure2.shape, "The two structures must have the same shape"
        assert isinstance(structure1, np.ndarray), 'Input structure1 must be a numpy array'
        assert isinstance(structure2, np.ndarray), 'Input structure2 must be a numpy array'
        assert isinstance(voxelSizes, np.ndarray), 'Input voxel_sizes must be a numpy array'
        # Compute the Euclidian distance transform, normal distance - distance inside structure
        dist = sndim.distance_transform_edt(1 - structure1, sampling=voxelSizes) - sndim.distance_transform_edt(sndim.binary_erosion(structure1, np.ones((3, 3, 3))), sampling=voxelSizes)  
        contour2 = structure2 ^ sndim.binary_erosion(structure2, np.ones((3, 3, 3)))
        distances = contour2 * dist
        return distances

  
    def saveDictToCSV(self, dataDict, observer, patient, structure, step, description):
        """
        Write a dictionary to a CSV file with comma as the delimiter.

        Args:
            dataDict (dict): Dictionary to be written
            observer (str): Observer identifier
            patient (str): Patient identifier
            structure (str): Structure identifier
            step (int): Step number
            description (str): Description of the file
        """
        # Assert input
        assert isinstance(dataDict, dict), 'Input dataDict must be a dictionary'
        assert isinstance(observer, str), 'Observer must be a string'
        assert isinstance(patient, str), 'Patient must be a string'
        assert isinstance(structure, str), 'Structure must be a string'
        assert isinstance(step, int), 'Step must be an integer'
        assert isinstance(description, str), 'Description must be a string'
        # Define file path
        filePath = os.path.join(conf.base.evalDataEditedOutFolderPath, f"{patient}_{observer}_{structure}_step{step}_{description}.csv")
        # Open the file
        with open(filePath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Loop through the dictionary and write to CSV
            for key, value in dataDict.items():
                writer.writerow([key, value])
        

    def calcSegMetrics(self, refData, obsData):
        """
        Calculate multiple segmentation metrics between reference and observer data.

        Args:
            refData (sitk object): Reference data.
            obsData (sitk object): Observer data.

        Returns:
            dict: A dictionary containing the segmentation metrics.
        """
        # Assert sitk objects
        assert isinstance(refData, sitk.Image), 'Input refData must be a SimpleITK object'
        assert isinstance(obsData, sitk.Image), 'Input obsData must be a SimpleITK object'
        # Cast as unsigned 8 bit 
        refData = sitk.Cast(refData, sitk.sitkUInt8)
        obsData = sitk.Cast(obsData, sitk.sitkUInt8)
        # Assert same image resolution
        assert refData.GetSpacing() == obsData.GetSpacing(), 'Input refData and obsData must have the same image resolution'
        # Assume isotropic resolution in-plane
        assert refData.GetSpacing()[0] == refData.GetSpacing()[1], 'Input refData and obsData must have isotropic resolution in-plane'
        assert obsData.GetSpacing()[0] == obsData.GetSpacing()[1], 'Input refData and obsData must have isotropic resolution in-plane'
        # Transaxial slices are assumed to be in the z-dimension (axis=2).
        # Assert 88 slices in the z-dimension
        assert refData.GetSize()[2] == conf.eval.numberTotalSlices, 'Input refData must have 88 slices in the z-dimension'
        assert obsData.GetSize()[2] == conf.eval.numberTotalSlices, 'Input obsData must have 88 slices in the z-dimension'
        # Compute volume
        volumeRefData = compute_volume(refData)
        volumeObsData = compute_volume(obsData)
        # Compute volume ratio
        volumeRatio = volumeObsData / volumeRefData
        # Compute volume difference
        volumeDifference = volumeRefData - volumeObsData
        # Compute the added path length for each slice where structure exists.
        # This is in voxels, not mm as claimed in the function. Verified by code authors. 
        apl_voxels = compute_apl(refData, obsData, conf.eval.apl_distance_threshold)
        # Convert to mm
        apl = [element * np.mean(refData.GetSpacing()[:2]) for element in apl_voxels] # same inplane resolution is required
        # Compute the mean (slice-wise) added path length in mm.
        # This operates on transaxial slices, which are assumed to be in the z-dimension (axis=2).
        apl_mean = compute_metric_mean_apl(refData, obsData, conf.eval.apl_distance_threshold)
        # Compute the total added path length in mm.
        # This operates on transaxial slices, which are assumed to be in the z-dimension (axis=2).
        apl_total = compute_metric_total_apl(refData, obsData, conf.eval.apl_distance_threshold)
        # Calculate sensitivity and specificity
        sensitivity = compute_metric_sensitivity(refData, obsData)
        specificity = compute_metric_specificity(refData, obsData)

        # Decapricated metrics
        # Compute the Dice similarity coefficient. Using Deep Mind surface_distance package.
        # dsc = compute_metric_dsc(refData, obsData, auto_crop=True)
        # Compute the Hausdorff distance. 
        # Choose to not use platipy as it is not validated.
        # hd_platipy = compute_metric_hd(refData, obsData, auto_crop=True)
        # Compute Surface Dice. 
        # Choose to not use platipy as it is not validated. 
        # surfaceDsc = compute_surface_dsc(refData, obsData, conf.eval.surface_dice_threshold)
        # Compute the mean absolute surface distance. 
        # Choose to not use platipy as it is not validated.
        # masd = compute_metric_masd(refData, obsData)

        # Calculate metrics using Deep Mind surface_distance package
        # Metrics in that package are validated by unit tests and expected values. 
        # Define variables used for the package 
        ref = sitk.GetArrayFromImage(refData).astype(bool)
        obs = sitk.GetArrayFromImage(obsData).astype(bool)
        spacing = refData.GetSpacing() # This is defined like row, column, slices 
        # Assert that last dimension equals expected slice thickness
        assert spacing[2] == conf.eval.sliceThickness, 'Slice thickness is not correct'
        # Transpose GT and pred so that the slices are in the last dimension
        ref = np.transpose(ref, (1,2,0))
        obs = np.transpose(obs, (1,2,0))
        # Assert that slice dimension is correct
        assert ref.shape[2] == conf.eval.numberTotalSlices, 'Number of slices is not correct'
        assert obs.shape[2] == conf.eval.numberTotalSlices, 'Number of slices is not correct'
        # Calculate surface distances (needs correct order of data and dimension spacing)
        surface_distances = surface_distance.compute_surface_distances(ref, obs, spacing_mm=spacing)
        # Calculate average surface distance. Two floats are returned:  
        # The average distance (in mm) from the ground truth surface to the
        # predicted surface. The average distance from the predicted surface to the ground truth
        # surface.
        asd_ref2obs, asd_obs2ref = surface_distance.compute_average_surface_distance(surface_distances)
        # Calculate surface DICE
        surfaceDsc = surface_distance.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=conf.eval.surface_dice_threshold)
        # Compute surface overlap (two floats returned here also for ref to obs and obs to ref)
        surfaceOverlap_ref2obs, surfaceOverlap_obs2ref = surface_distance.compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm=conf.eval.surface_overlap_threshold)
        # Calculate DICE
        dsc = surface_distance.compute_dice_coefficient(ref, obs)
        # Calculate robust HD, HD95 and HD99 (percentiles of HD)
        hd = surface_distance.compute_robust_hausdorff(surface_distances, 100)
        hd95 = surface_distance.compute_robust_hausdorff(surface_distances, 95)
        hd99 = surface_distance.compute_robust_hausdorff(surface_distances, 99)

        # Create dictionary
        metrics = {
            "VolumeRef": volumeRefData,
            "VolumeObs": volumeObsData,
            "VolumeRatio": volumeRatio,
            "VolumeDifference": volumeDifference,
            "MeanAPL": apl_mean,
            "TotalAPL": apl_total,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "DSC": dsc,
            "HD": hd,
            "HD95": hd95,
            "HD99": hd99,
            "SurfaceDSC": surfaceDsc,
            "SurfaceOverlap_ref2obs": surfaceOverlap_ref2obs,
            "SurfaceOverlap_obs2ref": surfaceOverlap_obs2ref,
            "ASD_ref2obs": asd_ref2obs,
            "ASD_obs2ref": asd_obs2ref,
            "APL": apl
            }
        # Return the metrics
        return metrics
    

    def DicomRT2Nifti(self, subjectStructFilePath, inputDicomReferenceFolderPath, outputFolderPath, observer, patient, structure, step, dataType):
        """
        Convert subject DICOM RT struct file to Nifty format

        Args:
            subjectStructFilePath (str): Path to the subject RT structure file
            inputDicomReferenceFolderPath (str): Path to the input DICOM reference folder
            outputFolderPath (str): Path to the output folder where the Nifty files will be saved    
            observer (str): Observer name
            patient (str): Patient name
            structure (str): Structure name
            step (int): Step number
            dataType (str): Data type (e.g. obsData or refData)        
            
        Returns:
            Outputs Nifti data to directory 
            newFilePath (str): Path to the new Nifti file

            
        """
        # Assess input
        assert isinstance(subjectStructFilePath, str), 'Input subjectStructFilePath must be a string'
        assert isinstance(inputDicomReferenceFolderPath, str), 'Input inputDicomReferenceFolderPath must be a string'
        assert isinstance(outputFolderPath, str), 'Input outputFolderPath must be a string'
        assert isinstance(observer, str), 'Input observer must be a string'
        assert isinstance(patient, str), 'Input patient must be a string'
        assert isinstance(structure, str), 'Input structure must be a string'
        assert isinstance(step, int), 'Input step must be an integer'
        assert isinstance(dataType, str), 'Input dataType must be a string'
        # Assert existing directories for output
        assert os.path.isdir(outputFolderPath), 'Input subjectStructFilePath must be a directory'
        # Get list of all structures present in the DICOM structure file 
        subjectStructList = list_rt_structs(subjectStructFilePath)
        # Count number of structures
        nrStructsinList = len(subjectStructList)
        # There should only be one structure
        assert nrStructsinList == 1, 'There should only be one structure in the list'
        # Convert the RT structure to Nifti format 
        # This is performed by targeting each individual structure at a time in a loop. 
        # This is slower but safer. 
        # In this way we can isolate exceptions to individual structures and not break 
        # the process of dcmrtstruc2nii which happens otherwise. This avoids modification of the 
        # dcmrtstruc2nii source code and allows us in retrospect to see if missing data was important or not.
        # Failed objects are often due to the fact that the structures are not completed or simply empty in Eclipse. 
        for structNr, currStruct in enumerate(subjectStructList):
            try:
                # Extract the structure and convert to Nifty
                # We do not want convert_original_dicom=True f as this will add a lot of compute time. 
                dcmrtstruct2nii(subjectStructFilePath, inputDicomReferenceFolderPath, outputFolderPath, structures=currStruct, gzip=True, mask_background_value=0, mask_foreground_value=1, convert_original_dicom=False)
            except:
                print("Exception when extracting " + currStruct)
                # Throw exception
                raise Exception('Exception when extracting ' + currStruct)
        
        # Get total number of files outputed
        nrFiles = len(os.listdir(outputFolderPath))
        # Assert that the number of files is equal to the number of structures
        assert nrFiles == nrStructsinList, 'Number of files outputted does not match the number of structures in the RT file'
        # Get the file name
        niftiFileName = os.listdir(outputFolderPath)[0]
        # Rename the file 
        if dataType == 'obsData': # If observer data
            newFilePath = os.path.join(outputFolderPath, 'mask_' + observer + '_' + patient + '_' + structure + '_' + 'step' + str(step) + '.nii.gz')
            os.rename(os.path.join(outputFolderPath, niftiFileName), newFilePath)
        if dataType == 'refData': # If reference data
            newFilePath = os.path.join(outputFolderPath, 'mask_' + patient + '_' + structure + '_ref.nii.gz')
            os.rename(os.path.join(outputFolderPath, niftiFileName), newFilePath)
        
        # Return the new file path
        return newFilePath


    def getRTStructFile(self, path):
        """
        Search a given path for a RT structure DICOM file
        Inputs:
            path (str): Path to the DICOM file directory
        Returns:
            The RT file name
        """
        # Assert input
        assert isinstance(path, str), 'Input path must be a string'
        # Assert directory
        assert os.path.isdir(path), 'Input path must be a directory'
        # List files 
        files = os.listdir(path)
        # Get only the RS struct dicom file 
        structFile = [f for f in files if ".dcm" in f]
        # Check that there is only one 
        if len(structFile) == 0:
            raise Exception('No RT structure file could be located. Make sure the file is located in the specified folder...')
        assert len(structFile) == 1 , 'There should only be one RT structure file in the folder'
        # Return data 
        return structFile[0]
    

    def readNiftiFile(self, filePath, dataType): 
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
        # Convert to datatype as defined in input
        if dataType == 'float64':
            np_imageData = np_imageData.astype(np.float64)
        if dataType == 'float32':
            np_imageData = np_imageData.astype(np.float32)
        if dataType == 'uint8':
            np_imageData = np_imageData.astype(np.uint8)
        if dataType == 'int8':
            np_imageData = np_imageData.astype(np.int8)
        # Return np_imagedata, sitk_imageData and pixel spacing
        return np_imageData, sitk_imageData, pixelSpacing
    

    def assertOnlyOneNiftiFile(self, folderPath):
        """
        Assert that there is only one Nifti file in the folder.

        Args:
            folderPath (str): Path to the folder
        """
        # Assert input
        assert isinstance(folderPath, str), 'Input folderPath must be a string'
        # Assert directory
        assert os.path.isdir(folderPath), 'Input folderPath must be a directory'
        # List files
        files = os.listdir(folderPath)
        # Get only the Nifti file
        niftiFiles = [f for f in files if ".nii.gz" in f]
        # Check that there is only one
        assert len(niftiFiles) == 1, 'There should only be one Nifti file in the folder'


    def calcSTAPLE(self, stackedSeg, threshold):
        """
        The STAPLE filter implements the Simultaneous Truth and Performance
        Level Estimation algorithm for generating ground truth volumes from a
        set of binary segmentations.

        arg: stackedSeg: 4D numpy array with shape (x, y, z, n) where n is the number of segmentations, 
        z is the number of slices and x and y are the number of voxels in the x and y direction.

        """
        # Assert 4D input
        assert len(stackedSeg.shape) == 4, 'Input stackedSegmentations must be 4D data'
        # Loop through last dimension, convert to SITK and stack in list
        segStackSitk = []
        for i in range(stackedSeg.shape[3]):
            # Convert to SITK object
            segSitk = sitk.GetImageFromArray(stackedSeg[:,:,:,i])
            segStackSitk.append(segSitk)
        # Run STAPLE algorithm
        SegSTAPLESitk = sitk.STAPLE(segStackSitk, 1.0 ) # 1.0 specifies the foreground value
        # Convert back to numpy array
        SegSTAPLE = sitk.GetArrayFromImage(SegSTAPLESitk)

        # Threshold the STAPLE segmentation
        SegSTAPLE[SegSTAPLE < threshold] = 0
        SegSTAPLE[SegSTAPLE >= threshold] = 1
        # Convert to int8
        SegSTAPLE = SegSTAPLE.astype(np.int8)
        # Return the STAPLE segmentation
        return SegSTAPLE


    def saveMatrixToNpz(self, dataMatrix, observer, patient, structure, step, description):
        """
        Save matrix to npz file. Will overwrite existing file.

        Args:
            matrix (np.array): Matrix to be saved
            filePath (str): Path to the file
        """
        # Assert input
        assert isinstance(dataMatrix, np.ndarray), 'Input matrix must be a numpy array'
        assert isinstance(observer, str), 'Input observer must be a string'
        assert isinstance(patient, str), 'Input patient must be a string'
        assert isinstance(structure, str), 'Input structure must be a string'
        assert isinstance(step, int), 'Input step must be an integer'
        assert isinstance(description, str), 'Input description must be a string'
        # Define file path 
        filePath = os.path.join(conf.base.evalDataEditedOutFolderPath, patient + '_' + observer + '_' + structure + '_' + f'step{step}' + '_' + description + '.npz')
        # Make sure folders exist
        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        # Save the matrix
        np.savez_compressed(filePath, data=dataMatrix)


    def loadNpzToMatrixAnyFormat(self, observer, patient, structure, step, description):
        """
        Load matrix from npz file. Does not required int8 format.

        Args:
            filePath (str): Path to the file

        Returns:
            np.array: Matrix loaded from the file
        """
        # Assert input
        assert isinstance(observer, str), 'Input observer must be a string'
        assert isinstance(patient, str), 'Input patient must be a string'
        assert isinstance(structure, str), 'Input structure must be a string'
        assert isinstance(step, int), 'Input step must be an integer'
        assert isinstance(description, str), 'Input description must be a string'
        # Define file path
        filePath = os.path.join(conf.base.evalDataEditedOutFolderPath, patient + '_' + observer + '_' + structure + '_' + f'step{step}' + '_' + description + '.npz')
        # Load the matrix
        data = np.load(filePath)['data']
        # Return the matrix
        return data
    

    def loadNpzToMatrix(self, observer, patient, structure, step, description):
        """
        Load matrix from npz file.

        Args:
            filePath (str): Path to the file

        Returns:
            np.array: Matrix loaded from the file
        """
        # Assert input
        assert isinstance(observer, str), 'Input observer must be a string'
        assert isinstance(patient, str), 'Input patient must be a string'
        assert isinstance(structure, str), 'Input structure must be a string'
        assert isinstance(step, int), 'Input step must be an integer'
        assert isinstance(description, str), 'Input description must be a string'
        # Define file path
        filePath = os.path.join(conf.base.evalDataEditedOutFolderPath, patient + '_' + observer + '_' + structure + '_' + f'step{step}' + '_' + description + '.npz')
        # Load the matrix
        data = np.load(filePath)['data']
        # Assert int 8 format
        assert data.dtype == np.int8, 'Input data must be int8 data'
        # Return the matrix
        return data


    def saveMatrixToNifti(self, dataMatrix, observer, patient, structure, step, description):
        """
        Save matrix to Nifti file. Will overwrite existing file.
        """
        # First an affine matrix for an existing geometry is needed.
        # This is independent of observer, therefore using obsB
        observerRef = 'obsB'

        # Define file path for reference Nifti file
        nifti_ref_path = os.path.join(conf.base.obsDataEditedFolderPath, observerRef, patient + '_' + observerRef, f'step{step}Edited', f'{structure}{step}', 'Nifti', f'mask_{observerRef}_{patient}_{structure}_step{step}.nii.gz')
        # Read with SimpleITK
        ref_sitk = sitk.ReadImage(nifti_ref_path)
        # Define file path for new Nifti file
        filePath = os.path.join(conf.base.evalDataEditedOutFolderPath, patient + '_' + observer + '_' + structure + '_' + f'step{step}' + '_' + description + '.nii.gz')
        # Make sure folders exist
        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        # Inherit the affine matrix from the ref image
        source_origin = ref_sitk.GetOrigin()
        source_direction = ref_sitk.GetDirection()
        source_spacing = ref_sitk.GetSpacing()
        # Reorder so slices in the 3D stack will be in the first dimension
        dataMatrix = np.transpose(dataMatrix, (2,0,1))
        # Convert matrix to sitk object
        sitk_image = sitk.GetImageFromArray(dataMatrix)
        # Set the origin, direction and spacing
        sitk_image.SetOrigin(source_origin)
        sitk_image.SetDirection(source_direction)
        sitk_image.SetSpacing(source_spacing)
        # Write the image
        sitk.WriteImage(sitk_image, filePath)


    def assertBinaryInt8Data(self, data): 
        """
        Assert that the data is binary.

        Args:
            data (np.array): Data array
        """
        # Assert input
        assert isinstance(data, np.ndarray), 'Input data must be a numpy array'
        # Assert int8 data
        assert data.dtype == np.int8, 'Input data must be int8 data'
        # Assert all values are either 0 or 1
        assert np.all(np.isin(data, [0, 1])), 'Input data must only contain values 0 or 1'


    def calcSegDifference(self, refData, obsData, assertDiffFlag=False, absFlag=False):
        """
        Calculate the segmentation difference between reference and observer data.

        Args:
            refData (numpy array): Reference data (can be observer data also).
            stepData (numpy array): Observer data.

        Returns:
            numpy array: Difference between reference and observer data.
        """
        # Assert binary data for refData and obsData
        self.assertBinaryInt8Data(refData)
        self.assertBinaryInt8Data(obsData)
        # Assert input
        if assertDiffFlag == True: 
            # Assert that the difference between two data arrays is not zero.
            # However, they must not be different if structure from AI is perfect. 
            self.assertDataDifference(refData, obsData)
        # Calculate the difference
        diffData = refData - obsData
        # If we are intested in only change of voxels without direction (add, remove) we can take the absolute value
        if absFlag == True:
            diffData = np.abs(diffData)
        # Return the difference
        return diffData


    def plotMiddleSlice(self, data,):
        """
        Plot the middle slice of a 3D data array.

        Args:
            data (np.array): Data array
            title (str): Title of the plot
        """
        # Assert input
        assert isinstance(data, np.ndarray), 'Input data must be a numpy array'
        # Assert 3D data
        assert len(data.shape) == 3, 'Input data must be 3D data'
        # Get the middle slice
        middleSlice = data[:,:,int(data.shape[2]/2)]
        # Plot the middle slice
        plt.imshow(middleSlice)
        plt.show()


    def convertDiffToNiftiLabels(self, diffData, label):
        """
        Convert difference data to Nifti data.
        Difference might be negative or positive.
        Move the negative part to equal a label of 2.
        """
        # Assert input
        assert isinstance(diffData, np.ndarray), 'Input diffData must be a numpy array'
        assert np.max(diffData) <= 1, 'Input diffData must 1 or less'
        assert np.min(diffData) >= -1, 'Input diffData must be -1 or larger'
        # Set value -1 to equal label
        diffData[diffData == -1] = label
        # Assess output, make sure min is 0 and max is label
        assert np.min(diffData) == 0, 'Output diffData must have min value 0'
        assert np.max(diffData) <= label, 'Output diffData must have max value 2'
        # Return the data
        return diffData


    def qaRTStruct(self, observer, patient, rtStructFilePath, step, dataType, containedStructure):
        """
        QA the read RT structure file. Make sure the patient ID is correct and that the structure name contains the containedStructure.
        This asserts that correct data has been exported and put to the correct folder. 
        This avoids mixup of wrong patient and wrong structure. 
        QA is also performed to check that reference data has not been modified in Eclipse before export. 
        This does however not differentiate between step 1 and step 2. However, it asserts that data must be changed in step 1 and step 2 compared to reference. 
        """
        assert isinstance(observer, str), 'Input observer must be a string'
        assert isinstance(patient, str), 'Input patient must be a string'
        assert isinstance(rtStructFilePath, str), 'Input rtStructFilePath must be a string'
        assert isinstance(step, int), 'Input step must be an integer'
        assert isinstance(dataType, str), 'Input dataType must be a string'
        assert isinstance(containedStructure, str), 'Input containedStructure must be a string'
        # Read file with pydicom. Simple ITK does not work for RT struct files.
        rtStruct = pydicom.dcmread(rtStructFilePath)
        # Get the structure set
        structureSet = rtStruct.StructureSetROISequence
        # Get the name of the first structure
        structureName = structureSet[0].ROIName
        # Create expected structure name
        if containedStructure == 'CTV':
            expectedStructureName = containedStructure + 'T' + '_427_10_AI'
        if containedStructure == 'Rectum':
            expectedStructureName = containedStructure + '_10_AI'
        # Get patient ID
        patientID = rtStruct.PatientID
        # Create expected patient ID from observer and patient
        expectedPatientID = patient + '_' + observer
        # Assert that the patient ID is correct
        assert patientID == expectedPatientID, 'The patient ID is not correct for ' + expectedPatientID
        # Assert that containedStructure is in the structureName
        assert structureName == expectedStructureName , 'The structure name for patient ' + patient + ' is not correct. It should be ' + expectedStructureName

        # As the reference structure is created with rt-utils originally we can check that the structure has not beed modified before export.
        # When a structure is modified in Eclipse the manufacturer tag is changed to "Varian Medical Systems".
        # If the structure is not modified the manufacturer tag is "Qurit".
        # Get manufacturer 
        manufacturer = rtStruct.Manufacturer
        # Check it 
        if dataType == 'ref':
            assert manufacturer == conf.eval.refDataManufacturer, 'ERROR: The manufacturer is not Qurit for reference data for patient ' + patient + ' and structure file ' + rtStructFilePath
        else:
            pass
                
        # Step 1
        if dataType == 'obs' and step == 1 and containedStructure == 'CTV' and patient + '_' + observer not in conf.eval.ignoreManufactQAStep1CTV:
            #pauseMe("The manufacturer is not Varian Medical Systems for observer data for patient " + patient + " and structure file " + rtStructFilePath + ". Press Enter to continue...")
            assert manufacturer == conf.eval.obsDataManufacturer, 'The manufacturer is not Varian Medical Systems for observer data for patient ' + patient + ' and structure file ' + rtStructFilePath

        if dataType == 'obs' and step == 1 and containedStructure == 'Rectum' and patient + '_' + observer not in conf.eval.ignoreManufactQAStep1Rectum:
            #pauseMe("The manufacturer is not Varian Medical Systems for observer data for patient " + patient + " and structure file " + rtStructFilePath + ". Press Enter to continue...")
            assert manufacturer == conf.eval.obsDataManufacturer, 'The manufacturer is not Varian Medical Systems for observer data for patient ' + patient + ' and structure file ' + rtStructFilePath
        
        # Step 2
        if dataType == 'obs' and step == 2 and containedStructure == 'CTV' and patient + '_' + observer not in conf.eval.ignoreManufactQAStep2CTV:
            #pauseMe("The manufacturer is not Varian Medical Systems for observer data for patient " + patient + " and structure file " + rtStructFilePath + ". Press Enter to continue...")
            assert manufacturer == conf.eval.obsDataManufacturer, 'The manufacturer is not Varian Medical Systems for observer data for patient ' + patient + ' and structure file ' + rtStructFilePath

        if dataType == 'obs' and step == 2 and containedStructure == 'Rectum' and patient + '_' + observer not in conf.eval.ignoreManufactQAStep2Rectum:
            #pauseMe("The manufacturer is not Varian Medical Systems for observer data for patient " + patient + " and structure file " + rtStructFilePath + ". Press Enter to continue...")
            assert manufacturer == conf.eval.obsDataManufacturer, 'The manufacturer is not Varian Medical Systems for observer data for patient ' + patient + ' and structure file ' + rtStructFilePath
  
 
    def sumDictColumns(self, dicts_list):
        """
        Sums the values for each key in a list of dictionaries.

        Args:
            dicts_list (list of dict): A list of dictionaries with numeric values.

        Returns:
            dict: A dictionary with the same keys, where the values are the sums of the values from the input dictionaries.
        """
        # Initialize an empty dictionary to store the sums
        summed_dict = {}
        # Iterate over each dictionary in the list
        for d in dicts_list:
            # Iterate over each key-value pair in the dictionary
            for key, value in d.items():
                # If the key is already in the summed_dict, add the current value to the existing value
                if key in summed_dict:
                    summed_dict[key] += value
                # If the key is not in the summed_dict, add it with the current value
                else:
                    summed_dict[key] = value
        # Retur the summed dictionary
        return summed_dict
    

    def assertDataDifference(self, data1, data2): 
        """
        Assert that the difference between two data arrays is not zero.

        Args:
            data1 (np.array): Data array 1
            data2 (np.array): Data array 2
        """
        # Assert input
        assert isinstance(data1, np.ndarray), 'Input data1 must be a numpy array'
        assert isinstance(data2, np.ndarray), 'Input data2 must be a numpy array'
        # Assert int8 data
        assert data1.dtype == np.int8, 'Input data1 must be int8 data'
        assert data2.dtype == np.int8, 'Input data2 must be int8 data'
        # Check if they are equal
        equalResult = np.array_equal(data1, data2)
        # Assert that the sum of the difference is not zero
        if equalResult == True:
            input("The compared matrixes are equal. Press Enter to continue...")


    def assertNiftiDataShape(self, ref_data, observer_data, ref_pxl_spacing, observer_pxl_spacing, structure):
        """
        Assert that the reference and observer data have the same shape and pixel spacing.

        Args:
            ref_data (np.array): Reference data
            observer_data (np.array): Observer data
            ref_pxl_spacing (tuple): Reference pixel spacing
            observer_pxl_spacing (tuple): Observer pixel spacing
            structure (str): Structure name
        """
        # assert ref_data.dtype == observer_data.dtype, f'{structure} reference and observer data have different data types'
        assert ref_data.shape == observer_data.shape, f'{structure} reference and observer data have different shapes'
        assert ref_pxl_spacing == observer_pxl_spacing, f'{structure} reference and observer data have different pixel spacing'


    def loadNiftiObserverData(self, observer, patient, step, structure, dataType):
        """
        Load Nifti observer data for a specific observer, patient, step, and structure.

        Args:
            observer (str): Observer name
            patient (str): Patient name
            step (int): Step number
            structure (str): Structure name

        Returns:
            tuple: A tuple containing the observer data, the SITK object, and the pixel spacing.
        """
        # Define file paths for observer data
        nifti_file_path = os.path.join(conf.base.obsDataEditedFolderPath, observer, patient + '_' + observer, f'step{step}Edited', f'{structure}{step}', 'Nifti', f'mask_{observer}_{patient}_{structure}_step{step}.nii.gz')
        # Print file path
        #print('Loading observer Nifti data from: ' + nifti_file_path)
        #print('Using variables :' + observer + ', ' + patient + ', ' + structure + ', ' + str(step))
        # Load observer Nifti data
        observer_data, observer_sitk, observer_pxl_spacing = self.readNiftiFile(nifti_file_path, dataType)
        # Return data
        return observer_data, observer_sitk, observer_pxl_spacing
    

    def loadNiftiSTAPLEData(self, observer, patient, step, structure, dataType):
        """
        Load Nifti STAPLE data for a specific patient, step, and structure.

        Args:
            patient (str): Patient name
            step (int): Step number
            structure (str): Structure name
            description (str): Description of the data

        Returns:
            tuple: A tuple containing the STAPLE data, the SITK object, and the pixel spacing.
        """
        # Assert input
        assert isinstance(observer, str), 'Input observer must be a string'
        assert isinstance(patient, str), 'Input patient must be a string'
        assert isinstance(structure, str), 'Input structure must be a string'
        assert isinstance(step, int), 'Input step must be an integer'
        assert isinstance(dataType, str), 'Input dataType must be a string'
        # Assert observer == obsAll for STAPLE
        assert observer == 'obsAll', 'Input observer must be obsAll'
        # Define file path 
        nifti_file_path = os.path.join(conf.base.evalDataEditedOutFolderPath, patient + '_' + observer + '_' + structure + '_' + f'step{step}' + '_' + 'staple' + '.nii.gz')
        # Load staple Nifti data
        staple_data, staple_sitk, staple_pxl_spacing = self.readNiftiFile(nifti_file_path, dataType)
        # Return the STAPLE data
        return staple_data, staple_sitk, staple_pxl_spacing


    def fakeSegDataSelectPieSlice(self, binaryMatrix, angle_rad):
        """
        Selects a pie-shaped part of the input 3D binary matrix using a given angle in radians.
        This allows us to create synthetic delineation data for testing purposes and thereby validate code used later in the analasis. 
        Specifically this can be used to assess that the plots look like expected when certain uncertainty voxcels are selected.
        Further, we can control the ratio of voxels between step 1 and step 2 and also make sure the same ratio of change 
        is seen in the plots. We assume symmetric distibution of uncertainty values around the center of mass of the binary matrix (reference data).
        Symmetry is broken for higher uncertainy values and this should be seen in the plots. 
        However, low uncertainty should follow the symmetry assumption with respect to pie angle. 
        Assumptions valid for prostate only. 
        
        Args: 
            param binaryMatrix: 3D numpy array (binary matrix)
            param angle_rad: Angle in radians for the pie slice
            param tolerance: Tolerance for the selected fraction of pixels

        Return: A mask of the same shape as binaryMatrix with the selected pie-shaped part
        """
        # Assert binary matrix
        self.assertBinaryInt8Data(binaryMatrix)
        # Get shape of the binary matrix
        rows, cols, slices = binaryMatrix.shape
        # Calculate the center of mass
        com = center_of_mass(binaryMatrix)
        # Get the center row, column, and slice as integers
        center_row, center_col, center_slice = map(int, com)
        # Create a grid of coordinates
        row_coords, col_coords, slice_coords = np.ogrid[:rows, :cols, :slices]
        # Calculate deltas relative to the center in 2D, 3D not needed 
        delta_row = row_coords - center_row
        delta_col = col_coords - center_col
        # Calculate the angle of each point relative to the center
        angles = np.arctan2(delta_row, delta_col)
        # Ensure the angles are in the range [0, 2*pi)
        angles = np.mod(angles, 2 * np.pi)
        # Create a mask for the pie slice
        pie_mask = (angles <= angle_rad)
        # Broadcast the mask to 3D by including all slices
        pie_mask_3d = np.broadcast_to(pie_mask, binaryMatrix.shape) 
        # Apply the mask to the input matrix
        outoutMatrix = binaryMatrix * pie_mask_3d
        # QA Calculate the fraction of selected pixels 
        total_pixels = np.sum(binaryMatrix[:,:,center_slice])
        selected_pixels = np.sum(outoutMatrix[:,:,center_slice])
        fraction_selected = selected_pixels / total_pixels
        # Print the fraction selected
        # print(f"Fraction selected: {fraction_selected:.4f}")
        # Return the output matrix
        return outoutMatrix
        

   
    def fakeSegDataNotUsed(self, refMatrix, uncMatrix, selFrac):
        """
        Create faked segmentation data based on the uncertainty matrix.
        Segmentation is selected from uncertainty voxels. 
        This allows us to create synthetic delineation data for testing purposes and thereby validate code used later in the analasis. 
        Specifically this can be used to assess that the plots look like expected when certain uncertainty voxcels are selected.
        Further, we can control the ratio of voxels between step 1 and step 2 and also make sure the same ratio of change 
        is seen in the plots. 
        
        Parameters:
        - self: reference to the class instance
        - refMatrix: numpy array (binary matrix)
        - uncertaintyMatrix: numpy array (matrix containing uncertainty values)
        - selFrac: float, selection fraction of voxels used for each uncertainty interval
        
        Returns:
        - fakeMatrix: numpy array with modified voxels
        """
        # Get all voxel indices as row, column where the uncertaintyMatrix falls within specified intervals
        # Intervals are 0.1-0.2, 0.2-0.3, 0.3-0.4, 0.4-0.5. Do not use lower than 0.1, a lot of voxels there. 
        array_01_02 = np.where((uncMatrix >= 0.1) & (uncMatrix < 0.2))
        array_02_03 = np.where((uncMatrix >= 0.2) & (uncMatrix < 0.3))
        array_03_04 = np.where((uncMatrix >= 0.3) & (uncMatrix < 0.4))
        array_04_05 = np.where((uncMatrix >= 0.4) & (uncMatrix < 0.5)) # Excludes large block of == 0.5 values
        # Convert the output of np.where to lists of (row, column, slice) tuples
        coords_01_02 = list(zip(array_01_02[0], array_01_02[1], array_01_02[2]))
        coords_02_03 = list(zip(array_02_03[0], array_02_03[1], array_02_03[2]))
        coords_03_04 = list(zip(array_03_04[0], array_03_04[1], array_03_04[2]))
        coords_04_05 = list(zip(array_04_05[0], array_04_05[1], array_04_05[2]))
        # Randomly select a fraction of the coordinates, these are amounts of voxels
        numCoords_01_02 = int(selFrac * len(coords_01_02))
        numCoords_02_03 = int(selFrac * len(coords_02_03))
        numCoords_03_04 = int(selFrac * len(coords_03_04))
        numCoords_04_05 = int(selFrac * len(coords_04_05))
        # Randomly select the fraction of the coordinates
        randomCoords_01_02 = random.sample(coords_01_02, numCoords_01_02)
        randomCoords_02_03 = random.sample(coords_02_03, numCoords_02_03)
        randomCoords_03_04 = random.sample(coords_03_04, numCoords_03_04)
        randomCoords_04_05 = random.sample(coords_04_05, numCoords_04_05)
        # Create a zero matrix with same shape as the reference matrix
        fakeMatrix = np.zeros(refMatrix.shape, dtype=np.int8)
        # Set the selected voxels to 1
        for coord in randomCoords_01_02:
            fakeMatrix[coord] = 1
        for coord in randomCoords_02_03:
            fakeMatrix[coord] = 1
        for coord in randomCoords_03_04:
            fakeMatrix[coord] = 1
        for coord in randomCoords_04_05:
            fakeMatrix[coord] = 1
        # Return the fake matrix
        return fakeMatrix


