# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Class for converting data from DICOM to Nifti format
# *********************************************************************************

import os
import cv2
import csv
import sys
import numpy as np
import os.path
import pydicom
import nibabel as nib
import SimpleITK as sitk
import scipy
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
import shutil
from datetime import datetime, timedelta
import time 
from joblib import Parallel, delayed

import cupy as cp
from cupyx.scipy.ndimage import label as cupyx_label


from commonConfig import commonConfigClass
# Load configuration
conf = commonConfigClass() 


class convertDataMethodsClass:
    """
    Class describing functions needed for converting DICOM data to Nifti data or
    vice versa.
    """

    def __init__ (self):
        """
        Init function
        """
        pass

    def getStudyAnonSubjectName(self, patient, csv_file_path):
        """
        Function to find and return the anonomized study name in the test data given a patient name.
        
        Args:
        - patient (str): The patient name to search for in the CSV file.
        - csv_file_path (str): The path to the CSV file containing the mapping of patient names.
        Returns:
        - str: The corresponding anonomized study name if patient is found in the CSV; otherwise throw error.
        """
        with open(csv_file_path, mode='r') as file:
            # Create a CSV reader object to iterate through rows
            reader = csv.DictReader(file)
            # Iterate through each row in the CSV file
            for row in reader:
                # Check if the current row's folderName matches the input folder_name
                if row['folderName'] == patient:
                    # Return the newPatientName from the matched row
                    return row['newPatientName']



    def qaPublicData(self, patientFolder, patient, referenceCountTxt, missingDataExpected):
        """
        Perform QA on a patient's folder by comparing NIfTI files to a reference file ('image.nii.gz') 
        and ensuring the expected number of files is found. Qualities checked include:
        - Spatial dimensions (x, y, z)
        - Voxel sizes
        - Transformation matrix (s-form)
        - Spatial origin offsets (x, y, z)
        - S-row vectors (srow_x, srow_y, srow_z)

        For 3D data, only the first slice is compared. All checks are performed exactly, with no tolerance.

        Also checks that the expected number of NIfTI files and text files are present in the folder.
        Further, we will check that there is only one connected component in the mask.

        Parameters:
        patientFolder (str): Path to the patient's folder containing NIfTI files.
        referenceCountNifti (int): Expected number of NIfTI files in the folder.
        referenceCountTxt (int): Expected number of text files in the folder.

        Returns:
        None: Prints mismatches for attributes and the file count check results.
        """
        # Print the patient folder being processed
        #print(f"Processing folder: {patientFolder}")
        # Locate the reference file ('image.nii.gz')
        referenceFilePath = os.path.join(patientFolder, 'image.nii.gz')
        if not os.path.exists(referenceFilePath):
            raise FileNotFoundError(f"Reference file 'image.nii.gz' not found in {patientFolder}")
        
        # Load the reference file
        referenceNifti = nib.load(referenceFilePath)
        referenceHeader = referenceNifti.header

        # Define attributes to compare
        attributesToCompare = {
            "dim": lambda hdr: hdr["dim"][1:4],  # Spatial dimensions (x, y, z)
            "pixdim": lambda hdr: hdr["pixdim"][1:4],  # Voxel sizes
            "stoXyz": lambda hdr: hdr.get_sform(),  # Transformation matrix (s-form)
            "qoffsetx": lambda hdr: hdr["qoffset_x"],  # Spatial origin x offset
            "qoffsety": lambda hdr: hdr["qoffset_y"],  # Spatial origin y offset
            "qoffsetz": lambda hdr: hdr["qoffset_z"],  # Spatial origin z offset
            "srow_x": lambda hdr: hdr["srow_x"],  # S-row vector (x)
            "srow_y": lambda hdr: hdr["srow_y"],  # S-row vector (y)
            "srow_z": lambda hdr: hdr["srow_z"],  # S-row vector (z)
        }

        # Walk through all files and compare with the reference
        for root, _, files in os.walk(patientFolder):
            # if image_reg2MRI.nii.gz is found, skip the file, already asserted quality when created
            if 'image_reg2MRI.nii.gz' in files:
                # Remove the file from the list
                files.remove('image_reg2MRI.nii.gz')
            for file in files:
                if file.endswith('.nii.gz') and file != 'image.nii.gz':
                    filePath = os.path.join(root, file)

                    # Load current NIfTI file
                    currentNifti = nib.load(filePath)
                    currentHeader = currentNifti.header

                    # If not fiducial data or dose data 
                    if 'fiducial' not in file and 'dose' not in file and 'uncertainty' not in file :
                        # Assert only one connected component in the mask from currentNifti
                        # Load the mask data
                        maskData = currentNifti.get_fdata()
                        # Assume maskData is a NumPy array, move it to GPU
                        maskData_gpu = cp.array(maskData)
                        # Assert that total mask signal is not 0
                        assert cp.sum(maskData_gpu) > 0, f"Mask signal is 0 for {file} for patient {patientFolder}"
                        # Define the structure for connectivity (3x3x3 for full 26-connectivity)
                        structure = cp.ones((3, 3, 3))
                        # Perform connected component labeling on GPU
                        labeledArray_gpu, numFeatures = cupyx_label(maskData_gpu, structure=structure)
                        #if numFeatures is not 1
                        if numFeatures != 1:
                            # Print the number of connected components
                            print(f"Number of connected components in {file}: {numFeatures} for patient {patientFolder}")
                            # Just print the file path for the file
                            print(filePath)

                    # Compare each attribute in the list between the current and reference NIfTI headers
                    for attr, getter in attributesToCompare.items():
                        currentVal = getter(currentHeader)
                        referenceVal = getter(referenceHeader)
                        # React only if the attribute values do not match
                        if not np.array_equal(currentVal, referenceVal):
                            # If we are working with something that is not original dose throw an error if the attribute values do not match
                            if 'dose_original' not in file:
                                print(f"Mismatch in {attr} for {file}: {currentVal} != {referenceVal}")
                                raise ValueError(f"Mismatch in {attr} for {file}: {currentVal} != {referenceVal}")
                            else: 
                                pass
                            
        # Define all expected Nifti files in the folder
        allStructs = ['image_reg2MRI.nii.gz', 'image.nii.gz', 'dose_original.nii.gz', 'dose_interpolated.nii.gz', 'mask_Genitalia.nii.gz', 'mask_Bladder.nii.gz', 'mask_PenileBulb.nii.gz', 'mask_CTVT_427.nii.gz', 'mask_FemoralHead_R.nii.gz', 'mask_PTVT_427.nii.gz', 'mask_FemoralHead_L.nii.gz', 'mask_BODY.nii.gz', 'mask_Rectum.nii.gz']
        
        # List all Nifti files in the patient folder and make sure they exist in the reference list
        niftiFilesForPatient = [f for f in os.listdir(patientFolder) if f.endswith('.nii.gz')]
        # Make sure that each file in the reference list is present in the patient folder. Except if folder path contains MR, then dose.nii.gz is not expected
        # Registered sCT to MRI is not expected either
        if 'MR_StorT2' in patientFolder:
            allStructs.remove('dose_original.nii.gz')
            allStructs.remove('image_reg2MRI.nii.gz')
        # Check if all files are present
        for struct in allStructs:
            if struct not in niftiFilesForPatient:
                # Check if patient is expected to have missing data
                if patient in missingDataExpected:
                    # Get the missing structures from dictionary
                    missingStructs = missingDataExpected[patient]
                    # Make sure missing structures are not in the patient data
                    for struct in missingStructs:
                        if struct in os.listdir(patientFolder):
                            raise ValueError(f"  Structure exists but is listed as missing: {struct}")
                else:
                    raise ValueError(f"  Missing structure: {struct}")
                
   
        # Count all txt files in the folder, do not walk
        txtFileCount = sum(1 for file in os.listdir(patientFolder) if file.endswith('.txt'))
        # print(f"Total txt files found: {txtFileCount}")
        if txtFileCount != referenceCountTxt:
            print(f"  Mismatch in file count! Expected: {referenceCountTxt}, Found: {txtFileCount}")
            raise ValueError(f"  Mismatch in file count! Expected: {referenceCountTxt}, Found: {txtFileCount} for patient" + patientFolder)
           
        

    def getRTDoseFile(self, path):
        """
        Search a given path for a RT dose DICOM file
        Inputs:
            path (str): Path to the DICOM file directory
        Returns:
            The RT dose file name
        """
        # Assert input
        assert isinstance(path, str), 'Input path must be a string'
        # Assert directory
        assert os.path.isdir(path), 'Input path must be a directory'
        # List files 
        files = os.listdir(path)
        # Get only the dose struct dicom file 
        doseFile = [f for f in files if ".dcm" in f]
        # Check that there is only one 
        if len(doseFile) == 0:
            raise Exception('No RT dose file could be located. Make sure the file is located in the specified folder...')
        assert len(doseFile) == 1
        # Return data 
        return doseFile[0]
    

    def DicomFiducial2Nifti(self, i_subject, subject, dataInBasePath, dataInBasePathNifti, dataOutBasePath, imageFolder, contourFolder): 
        """
        Convert subject DICOM fiducial data to Nifty format and text file

        Args: 
            i_subject (int): The current subject number
            subject (str): The current subject name
            dataInBasePath (str): The base path to the DICOM dataset
            dataOutBasePath (str): The base path to the Nifti dataset
            fiducialFolder (str): The folder where the fiducial data is located
            imageFolder (str): The folder where the image data is located
        """
        # Assess input
        assert isinstance(i_subject, int), 'Input i_subject must be an integer'
        assert isinstance(subject, str), 'Input subject must be a string'
        assert isinstance(dataInBasePath, str), 'Input dataInBasePath must be a string'
        assert isinstance(dataOutBasePath, str), 'Input dataInBasePath must be a string'
        assert isinstance(contourFolder, str), 'Input doseFolder must be a string'
        # Assert existing directories
        assert os.path.isdir(dataInBasePath), 'Input dataInBasePath must be a directory'
        os.makedirs(dataOutBasePath, exist_ok=True)
        assert os.path.isdir(dataOutBasePath), 'Input dataOutBasePath must be a directory'
        # Get the RT fiducial struct file and path 
        subjectFolderPath = os.path.join(dataInBasePath, subject)
        subjectFuducialStructFile = self.getRTStructFile(os.path.join(subjectFolderPath, conf.preProcess.RTstructFolder, contourFolder))
        subjectFiducialStructFilePath = os.path.join(subjectFolderPath, conf.preProcess.RTstructFolder, contourFolder, subjectFuducialStructFile)
        # Get the Nifti image volume file and path
        subjectNiftiImageFilePath = os.path.join(dataInBasePathNifti, subject, 'image' + '.nii.gz')
        # Define subject output folder
        subjectOutFolderPath = os.path.join(dataOutBasePath, subject, imageFolder)
        os.makedirs(subjectOutFolderPath, exist_ok=True)
        from convertRTFiducialStruct import RTFiducialProcessor
        # Create an instance of RTFiducialProcessor and process the files
        processor = RTFiducialProcessor(subjectFiducialStructFilePath, subjectNiftiImageFilePath, subjectOutFolderPath)
        # Convert the RT fiducial struct file to Nifty format
        processor.convert_rtstruct()
     

    def regsCT2MRIForPublic(self, i_subject, subject, dataOutBasePath, sCTFolder, MRIFolder):
        """
        Register sCT to MRI data for public repository

        Args:
            i_subject (int): The current subject number
            subject (str): The current subject name
            dataOutBasePath (str): The base path to the Nifti dataset
            sCTFolder (str): The folder where the sCT data is located
            MRIFolder (str): The folder where the MRI data is located
        """
        # Assess input
        assert isinstance(i_subject, int), 'Input i_subject must be an integer'
        assert isinstance(subject, str), 'Input subject must be a string'
        assert isinstance(dataOutBasePath, str), 'Input dataInBasePath must be a string'
        assert isinstance(sCTFolder, str), 'Input sCTFolder must be a string'
        assert isinstance(MRIFolder, str), 'Input MRIFolder must be a string'
        # Assert existing directories
        os.makedirs(dataOutBasePath, exist_ok=True)
        assert os.path.isdir(dataOutBasePath), 'Input dataOutBasePath must be a directory'
        # sCT file path 
        sCTFilePath = os.path.join(dataOutBasePath, subject, sCTFolder, 'image' + '.nii.gz')
        # MRI file path
        MRIFilePath = os.path.join(dataOutBasePath, subject, MRIFolder, 'image' + '.nii.gz')
        # Define output path for the registered sCT to MRI
        outputPathReg = os.path.join(dataOutBasePath, subject, sCTFolder, 'image_reg2MRI' + '.nii.gz')
        # Determine if we use oldAcq or newAcq protocol
        acqValue = None
        if 'newAcq' in MRIFilePath and 'newAcq' in sCTFilePath: 
            acqValue = 'newAcq'
        if 'oldAcq' in MRIFilePath and 'oldAcq' in sCTFilePath: 
            acqValue = 'oldAcq'
        # Assert assigned either of them 
        assert acqValue is not None, "Something went wrong, could not determine the acquisition protocol."
        # Load images
        sCT_image = sitk.ReadImage(sCTFilePath)
        mri_image = sitk.ReadImage(MRIFilePath)
        # Set the first 50 rows in each slice of the CT image volume to -999.5.
        # This is to remove the sCT warning text existing on oldAcq images. 
        sCT_image_original_np = sitk.GetArrayFromImage(sCT_image)
        # Get the original matrix size and voxel size for sCT
        sCTMatrixSizeOrig = sCT_image_original_np.shape
        # Set number of slices last for shape
        sCTMatrixSizeOrig = (sCTMatrixSizeOrig[1], sCTMatrixSizeOrig[2], sCTMatrixSizeOrig[0])
        sCTVoxelSizeOrig = sCT_image.GetSpacing()   
        # Round to 4 decimals and convert to tuple
        sCTVoxelSizeOrig = np.round(sCTVoxelSizeOrig, 4)
        sCTVoxelSizeOrig = tuple(sCTVoxelSizeOrig)
        # Assert in plane resolution is the same in the sCT
        assert sCTVoxelSizeOrig[0] == sCTVoxelSizeOrig[1], "Voxel size mismatch between sCT in x and y direction."
        # Get the matrix size and voxel size for MRI
        mri_image_np = sitk.GetArrayFromImage(mri_image)
        mriMatrixSize = mri_image_np.shape
        # Set number of slices last for shape
        mriMatrixSize = (mriMatrixSize[1], mriMatrixSize[2], mriMatrixSize[0])
        mriVoxelSize = mri_image.GetSpacing()
        # Round to 4 decimals and convert to tuple
        mriVoxelSize = np.round(mriVoxelSize, 4)
        mriVoxelSize = tuple(mriVoxelSize)
        # Assert in plane resolution is the same in the MRI
        assert mriVoxelSize[0] == mriVoxelSize[1], "Voxel size mismatch between MRI in x and y direction."
        
        # Copy
        sCT_image_np = sCT_image_original_np.copy()
        # Remove text from oldAcq images
        if acqValue == 'oldAcq': 
                sCT_image_np[:, :50, :] = -999.5
                
        # Convert back to SimpleITK Image
        mod_sCT_image = sitk.GetImageFromArray(sCT_image_np)
        mod_sCT_image.CopyInformation(sCT_image)  # Retain the spatial info from original image
        # Register the sCT to MRI
        resample = sitk.ResampleImageFilter()
        # Set the reference image to the MRI
        resample.SetReferenceImage(mri_image)
        # Resample with bspline interpolation
        resample.SetInterpolator(sitk.sitkBSpline)
        #resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(-999.5) # Hounsfield unit for air
        resampled_image = resample.Execute(mod_sCT_image)
        # Edit datatype in Nifti header to make sure decimals are not truncated
        resampled_image.SetMetaData('datatype', '8')
        # Check matrix (size) dimensions between MRI and resampled CT
        assert mri_image.GetSize() == resampled_image.GetSize(), "Matrix size mismatch between MRI and resampled CT."
        # Check voxel size (spacing)
        assert mri_image.GetSpacing() == resampled_image.GetSpacing(), "Voxel size mismatch between MRI and resampled CT."
        # Check orientation (direction)
        assert mri_image.GetDirection() == resampled_image.GetDirection(), "Orientation mismatch between MRI and resampled CT."
        # Check spatial origin offsets
        assert mri_image.GetOrigin() == resampled_image.GetOrigin(), "Spatial origin offset mismatch between MRI and resampled CT."
        # Save the registered sCT to MRI
        sitk.WriteImage(resampled_image, outputPathReg)
        # Make sure file exists
        assert os.path.isfile(outputPathReg), "Registered sCT to MRI file not found, not written it seams."

        # Handle QA for dose data
        # Get file path for the original dose data
        origDoseFilePath = os.path.join(dataOutBasePath, subject, sCTFolder, 'dose_original' + '.nii.gz')
        # Load Nifti file for the original dose data
        origDoseImage = sitk.ReadImage(origDoseFilePath)
        # Get the original dose data as a numpy array
        origDoseImage_np = sitk.GetArrayFromImage(origDoseImage)
        # Get the original dose matrix size
        doseMatrixSizeOrig = origDoseImage_np.shape
        # Set number of slices last for shape
        doseMatrixSizeOrig = (doseMatrixSizeOrig[1], doseMatrixSizeOrig[2], doseMatrixSizeOrig[0])
        # Get the original dose voxel size 
        doseVoxelSizeOrig = origDoseImage.GetSpacing()
        # Round to 4 decimals and convert to tuple
        doseVoxelSizeOrig = np.round(doseVoxelSizeOrig, 4)
        doseVoxelSizeOrig = tuple(doseVoxelSizeOrig)
        # Get min and max dose values
        doseMin = origDoseImage_np.min()
        doseMax = origDoseImage_np.max()
        # Round to 2 decimals
        doseMin = np.round(doseMin, 2)
        doseMax = np.round(doseMax, 2)
        
        # Return matrix size and voxel size for sCT and MRI and dose data
        return sCTMatrixSizeOrig, sCTVoxelSizeOrig, mriMatrixSize, mriVoxelSize, doseMatrixSizeOrig, doseVoxelSizeOrig, doseMin, doseMax 
        
    
    def DicomDose2Nifti(self, i_subject, subject, dataInBasePath, dataOutBasePath, imageFolder, doseFolder):
        """
        Convert subject DICOM dose data to Nifty format

        Args:
            i_subject (int): The current subject number
            subject (str): The current subject name
            dataInBasePath (str): The base path to the DICOM dataset
            dataOutBasePath (str): The base path to the Nifti dataset
            imageFolder (str): The folder where the image data is located
            doseFolder (str): The folder where the dose data is located
        """
        # Assess input
        assert isinstance(i_subject, int), 'Input i_subject must be an integer'
        assert isinstance(subject, str), 'Input subject must be a string'
        assert isinstance(dataInBasePath, str), 'Input dataInBasePath must be a string'
        assert isinstance(dataOutBasePath, str), 'Input dataInBasePath must be a string'
        assert isinstance(imageFolder, str), 'Input imageFolder must be a string'
        assert isinstance(doseFolder, str), 'Input doseFolder must be a string'
        # Assert existing directories
        assert os.path.isdir(dataInBasePath), 'Input dataInBasePath must be a directory'
        os.makedirs(dataOutBasePath, exist_ok=True)
        assert os.path.isdir(dataOutBasePath), 'Input dataOutBasePath must be a directory'
        # Get the RT dose dicom file and path 
        subjectFolderPath = os.path.join(dataInBasePath, subject)
        subjectDoseFolderPath = os.path.join(subjectFolderPath, doseFolder)
        subjectDoseFile = self.getRTDoseFile(os.path.join(subjectFolderPath, subjectDoseFolderPath))
        subjectDoseFilePath = os.path.join(subjectDoseFolderPath, subjectDoseFile)
        # Define subject output folder
        subjectOutFolderPath = os.path.join(dataOutBasePath, subject, imageFolder)
        # Create the output folder
        os.makedirs(subjectOutFolderPath, exist_ok=True)
        # Read the dose image with sitk
        doseImageOriginal = sitk.ReadImage(subjectDoseFilePath)
        # Read file with pydicom again to get the dose grid scaling factor 
        doseDataPyDicom = pydicom.dcmread(subjectDoseFilePath)
        doseGridScaling = doseDataPyDicom.DoseGridScaling
        # Define the Nifti volume file path. This image volume is the out that is outputted. 
        subjectNiftiImageFilePath = os.path.join(subjectOutFolderPath, 'image' + '.nii.gz')
        # Read the Nifti image with sitk. We will inherit the orientation from this image.
        imageVolume = sitk.ReadImage(subjectNiftiImageFilePath)
        # Resample the dose image to the same orientation as the Nifti image
        doseImageResampled = sitk.Resample(doseImageOriginal, imageVolume,
                                            transform=sitk.Transform(),
                                            interpolator=sitk.sitkLinear,  # sitk.sitkBSpline,  # sitk.sitkLinear
                                            defaultPixelValue=0,
                                            outputPixelType=doseImageOriginal.GetPixelID(),
                                            )
        # Cast to float32 to be able to scale the dose appropriately
        doseImageResampled = sitk.Cast(doseImageResampled, sitk.sitkFloat32)
        doseImageOriginal = sitk.Cast(doseImageOriginal, sitk.sitkFloat32)
        # Scale the dose 
        doseImageResampled = (doseImageResampled * doseGridScaling)
        doseImageOriginal = (doseImageOriginal * doseGridScaling)

        # Assertions to verify the exact geometry match
        assert doseImageResampled.GetOrigin() == imageVolume.GetOrigin(), \
            f"Origin mismatch: {doseImageResampled.GetOrigin()} != {imageVolume.GetOrigin()}"
        assert doseImageResampled.GetSpacing() == imageVolume.GetSpacing(), \
            f"Spacing mismatch: {doseImageResampled.GetSpacing()} != {imageVolume.GetSpacing()}"
        assert doseImageResampled.GetDirection() == imageVolume.GetDirection(), \
            f"Direction mismatch: {doseImageResampled.GetDirection()} != {imageVolume.GetDirection()}"
        assert doseImageResampled.GetSize() == imageVolume.GetSize(), \
            f"Size mismatch: {doseImageResampled.GetSize()} != {imageVolume.GetSize()}"

        # Write the interpolated sitk image to Nifti format.
        # Define output Path for interpolated data 
        outputPathDoseInterpolated = os.path.join(subjectOutFolderPath, 'dose_interpolated.nii.gz')
        # Save the NIfTI image to the specified path
        sitk.WriteImage(doseImageResampled, outputPathDoseInterpolated)
        # Write a txt file with the dose grid scaling factor, min dose and max dose
        # Define output Path for dose data
        #outputPathDoseData = os.path.join(subjectOutFolderPath, 'doseData.txt')
        # Write the dose data to the txt file
        #with open(outputPathDoseData, 'w') as f:
        #    f.write('Dose grid scaling factor applied: ' + str(doseGridScaling) + '\n')
        #    f.write('Minimum resampled dose: ' + str(sitk.GetArrayFromImage(doseImageResampled).min()) + '\n')
        #    f.write('Maximum resampled dose: ' + str(sitk.GetArrayFromImage(doseImageResampled).max()) + '\n')


        # Write the original dose matrix to Nifti format.
        # Only if image folder is sCT
        if imageFolder == 'sCT':
            # Define output Path for original data
            outputPathDoseOriginal = os.path.join(subjectOutFolderPath, 'dose_original.nii.gz')
            # Save the NIfTI image to the specified path
            sitk.WriteImage(doseImageOriginal, outputPathDoseOriginal)
 

    def DicomRT2NiftiForPublic(self, i_subject, subject, dataInBasePath, dataOutBasePath, imageFolder, contourFolder):
        """
        Convert subject DICOM CT and struct data to Nifty format
        Used to convert data for public repository.
        Had to add some functionality and therefore did a separate function.

        Args:
            i_subject (int): The current subject number
            subject (str): The current subject name
            dataInBasePath (str): The base path to the DICOM dataset
            dataOutBasePath (str): The base path to the Nifti dataset
            
        Returns:
            Outputs data to directory 
            
        """
        # Assess input
        assert isinstance(i_subject, int), 'Input i_subject must be an integer'
        assert isinstance(subject, str), 'Input subject must be a string'
        assert isinstance(dataInBasePath, str), 'Input dataInBasePath must be a string'
        assert isinstance(dataOutBasePath, str), 'Input dataInBasePath must be a string'
        # Assert existing directories
        assert os.path.isdir(dataInBasePath), 'Input dataInBasePath must be a directory'
        os.makedirs(dataOutBasePath, exist_ok=True)
        assert os.path.isdir(dataOutBasePath), 'Input dataOutBasePath must be a directory'

        # Get the RT struct file and path 
        subjectFolderPath = os.path.join(dataInBasePath, subject)

        #subjectCTFolderPath = os.path.join(subjectFolderPath, conf.preProcess.CTfolder)
        subjectCTFolderPath = os.path.join(subjectFolderPath, imageFolder)
        subjectStructFile = self.getRTStructFile(os.path.join(subjectFolderPath, conf.preProcess.RTstructFolder, contourFolder))
        subjectStructFilePath = os.path.join(subjectFolderPath, conf.preProcess.RTstructFolder, contourFolder, subjectStructFile)
        
        # Define subject output folder
        subjectOutFolderPath = os.path.join(dataOutBasePath, subject, imageFolder)
        os.makedirs(subjectOutFolderPath, exist_ok=True)
        # Get list of all structures present in the DICOM structure file 
        subjectStructList = list_rt_structs(subjectStructFilePath)
        # Count number of structures
        nrStructsinList = len(subjectStructList)

        # Define list of wanted structures for the public repository
        # This is done to avoid the inclusion of unwanted structures in the public repository
        publicStructures = ['Bladder', 'Rectum', 'BODY', 
                            'CTVT_42.7', 'PTVT_42.7', 'FemoralHead_L', 'FemoralHead_R', 
                            'Genitalia', 'PenileBulb'] 

        # Filter subjectStructList with publicStructures
        subjectStructList = [x for x in subjectStructList if x in publicStructures]
        # Convert the RT structs to Nifty format 
        # This is performed by targeting each individual structure at a time in a loop. 
        # This is slower but safer. 
        # In this way we can isolate exceptions to individual structures and not break 
        # the process of dcmrtstruc2nii which happens otherwise. This avoids modification of the 
        # dcmrtstruc2nii source code and allows us in retrospect to see if missing data was important or not.
        # Failed objects are due to the fact that the structures are not completed or simply empty in Eclipse. 
        for structNr, currStruct in enumerate(subjectStructList):
            try:
                # Extract the structure and convert to Nifty
                # We do not want convert_original_dicom=True for all structures as this will add a lot of compute time. 
                # Do this only for BODY as this structure is always present. It has nothing to do with the structure itself for enabling convert_original_dicom=True. 
                if currStruct in conf.base.bodyStructureName1 or currStruct in conf.base.bodyStructureName2:
                    print(subject)
                    dcmrtstruct2nii(subjectStructFilePath, subjectCTFolderPath, subjectOutFolderPath, structures=currStruct, gzip=True, mask_background_value=0, mask_foreground_value=1, convert_original_dicom=True)
                else:
                    dcmrtstruct2nii(subjectStructFilePath, subjectCTFolderPath, subjectOutFolderPath, structures=currStruct, gzip=True, mask_background_value=0, mask_foreground_value=1, convert_original_dicom=False)
                
            except:
                print("Exception when extracting " + currStruct + ' for ' + subject )
        
        # Get total number of files outputted
        #nrFiles = len(os.listdir(subjectOutFolderPath))
        # If number of output files and in the list differ
        # -1 becuase of the image file that is created by dcmrtstruct2nii
        #if nrFiles -1 != nrStructsinList:
        #    # Throw message   
        #    print('Number of output files and in the list differ for patient ' + subject )
        #    print(str(int(nrStructsinList-(nrFiles -1))) + ' structures were not extracted')
        #    print(subjectStructList)
        #    print(os.listdir(subjectOutFolderPath))


    def DicomRT2Nifti(self, i_subject, subject, dataInBasePath, dataOutBasePath):
        """
        Convert subject DICOM CT and struct data to Nifty format

        Args:
            i_subject (int): The current subject number
            subject (str): The current subject name
            dataInBasePath (str): The base path to the DICOM dataset
            dataOutBasePath (str): The base path to the Nifti dataset
            
        Returns:
            Outputs data to directory 
            
        """
        # Assess input
        assert isinstance(i_subject, int), 'Input i_subject must be an integer'
        assert isinstance(subject, str), 'Input subject must be a string'
        assert isinstance(dataInBasePath, str), 'Input dataInBasePath must be a string'
        assert isinstance(dataOutBasePath, str), 'Input dataInBasePath must be a string'
        # Assert existing directories
        assert os.path.isdir(dataInBasePath), 'Input dataInBasePath must be a directory'
        os.makedirs(dataOutBasePath, exist_ok=True)
        assert os.path.isdir(dataOutBasePath), 'Input dataOutBasePath must be a directory'
        # Get the RT struct file and path 
        subjectFolderPath = os.path.join(dataInBasePath, subject)
        subjectCTFolderPath = os.path.join(subjectFolderPath, conf.preProcess.CTfolder)
        subjectStructFile = self.getRTStructFile(os.path.join(subjectFolderPath, conf.preProcess.RTstructFolder, 'sCTContours'))
        subjectStructFilePath = os.path.join(subjectFolderPath, conf.preProcess.RTstructFolder, 'sCTContours', subjectStructFile)
        # Define subject output folder
        subjectOutFolderPath = os.path.join(dataOutBasePath, subject)
        os.makedirs(subjectOutFolderPath, exist_ok=True)
        # Get list of all structures present in the DICOM structure file 
        subjectStructList = list_rt_structs(subjectStructFilePath)
        # Count number of structures
        nrStructsinList = len(subjectStructList)
        # Convert the RT structs to Nifty format 
        # This is performed by targeting each individual structure at a time in a loop. 
        # This is slower but safer. 
        # In this way we can isolate exceptions to individual structures and not break 
        # the process of dcmrtstruc2nii which happens otherwise. This avoids modification of the 
        # dcmrtstruc2nii source code and allows us in retrospect to see if missing data was important or not.
        # Failed objects are due to the fact that the structures are not completed or simply empty in Eclipse. 
        for structNr, currStruct in enumerate(subjectStructList):
            try:
                # Extract the structure and convert to Nifty
                # We do not want convert_original_dicom=True for all structures as this will add a lot of compute time. 
                # Do this only for BODY as this structure is always present. It has nothing to do with the structure itself for enabling convert_original_dicom=True. 
                if currStruct in conf.base.bodyStructureName1 or currStruct in conf.base.bodyStructureName2:
                    print(subject)
                    dcmrtstruct2nii(subjectStructFilePath, subjectCTFolderPath, subjectOutFolderPath, structures=currStruct, gzip=True, mask_background_value=0, mask_foreground_value=1, convert_original_dicom=True)
                else:
                    dcmrtstruct2nii(subjectStructFilePath, subjectCTFolderPath, subjectOutFolderPath, structures=currStruct, gzip=True, mask_background_value=0, mask_foreground_value=1, convert_original_dicom=False)
                
            except:
                print("Exception when extracting " + currStruct + ' for ' + subject )
        
        # Get total number of files outputted
        nrFiles = len(os.listdir(subjectOutFolderPath))
        # If number of output files and in the list differ
        # -1 becuase of the image file that is created by dcmrtstruct2nii
        if nrFiles -1 != nrStructsinList:
            # Throw message   
            print('Number of output files and in the list differ for patient ' + subject )
            print(str(int(nrStructsinList-(nrFiles -1))) + ' structures were not extracted')
            print(subjectStructList)
            print(os.listdir(subjectOutFolderPath))

        
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
        #structFile = [f for f in files if "RS" in f]
        # Check that there is only one 
        if len(structFile) == 0:
            raise Exception('No RT structure file could be located. Make sure the file is located in the specified folder...')
        assert len(structFile) == 1
        # Return data 
        return structFile[0]
    

    def readDicomSeries2SITK(self, input_directory_with_DICOM_series):
        """
        This function reads a DICOM series and returns a SimpleITK image object.
        """
        # Read the original DICOM series. First obtain the series file names using the
        # image series reader.
        data_directory = input_directory_with_DICOM_series
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory)
        if not series_IDs:
            print(
                'ERROR: given directory "'
                + data_directory
                + '" does not contain a DICOM series.'
            )
            # sys.exit(1)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            data_directory, series_IDs[0]
        )

        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)

        # Configure the reader to load all of the DICOM tags (public+private):
        # By default tags are not loaded (saves time).
        # By default if tags are loaded, the private tags are not loaded.
        # We explicitly configure the reader to load tags, including the
        # private ones. Be aware that not ALL private tags are supported.
        # Some are left out, like in GE PET images.  
        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()
        # Even though imag3D is not used it must be read like this to get the meta data
        image3D = series_reader.Execute()
        # Return image3D
        return image3D, series_reader

  
    def uncertaintyNifti2DICOM(self, patient, input_directory_with_DICOM_series, output_directory, uncertaintyMapFilePath):
        """
        Convert a NIfTI uncertainty map to a PET DICOM series. 
        The uncertainty map is a 3D NIfTI file with the same dimensions as the original MRI series. 
        Original MRI DICOM files from the same patient are loaded in the input_directory_with_DICOM_series.
        The output_directory is where the new DICOM series will be saved.
        This will be defined as a PET series in DICOM format. This allows for showing color in Eclipse as 
        Eclipse use color for PET images. These colors are added in Eclipse, not by any DICOM standard. 
        It is important to realize that the shown colors in Eclipse are determined by the values 
        of its color axis, seen when clicking the window button. 
        It uses auto windowing as default. This means that we have to create 
        conditions that provide good auto windowing. We therefore edit the background in the image to set the scale. 
        This is done by defining a box of 200 rows anterior to the patient anatomy. A lot of work went into
        this function, about 3-4 weeks before I got it right. Key was to multiply uncertainty with 10 000 so values
        could be maintained (divide by 100 to get std percent). Another reason for the long process was that
        I did not realize that PET scale could be manually set in Eclipse. This is done by clicking the window button
        and sliding the scale. For many cases I thought the input data was unfit when it was just the displayed scale that was wrong.
        I have verified that given values are correct between simple ITK and Eclipse. 
        """
        # Factor to multiply the scale value with to set the background value for the defined box. 
        # This sets the upper scale value. Needed for color scale consistency between patients.
        bgScaling = 0.5
        # Scaling of uncertainty to later be defined as signed int16
        scaleValue = 10000 #So we easy can read the uncertainty map in Eclipse, just divide by 100 to get percentage of std. 
        # Print message to console per patient
        print("Converting uncertainty map to DICOM for patient " + patient)
        # Load the uncertainty map from a NIfTI file
        uncertaintyMap = nib.load(uncertaintyMapFilePath).get_fdata()      
        # Print max value
        print('Max value in uncertainty map: ' + str(np.max(uncertaintyMap)))
        # Assert it to be less than bgScaling * scaleValue, otherwise we have to increase signal value in defined box 
        assert np.max(uncertaintyMap) <= bgScaling * scaleValue, 'Max value in uncertainty map must be less or equal to ' + str(bgScaling * scaleValue)
        assert np.min(uncertaintyMap) >= 0, 'Min value in uncertainty map must be larger or equal to 0'       
        # Convert to signed int16 (standard PET format). Scale value is 10000, makes it easy to read in Eclipse.
        uncertaintyMap = (uncertaintyMap * scaleValue).astype(np.int16) 
        # Define a block of 200 rows anterior to the patient anatomy 
        # having the value 0. This is to set the scale in Eclipse. (SITK format)
        # This is good for color scale consistency between patients.
        # Otherwise this must be handled manually by adjusting color scale slides in Eclipse.
        uncertaintyMap[:,0:200,:] = 0
        # A similar block with the highest value is placed posterior to the patient anatomy
        # Set last 200 rows of all slices to scaleValue*bgScaling. 
        uncertaintyMap[:,-200:,:] = scaleValue*bgScaling # So if bgScaling is 0.5, the max value will be 5000. This also will mean that the most red color of the spectra will equal 5000 (or 0.5 std).
        # Assert max value for color scale
        assert np.max(uncertaintyMap) == scaleValue*bgScaling, 'Max value in uncertainty map must be to set value in box'
        # Assert min value for color scale to be 0
        assert np.min(uncertaintyMap) == 0, 'Min value in uncertainty map must be 0, as in defined box'

        # Rotate for correct orientation into SITK format 
        uncertaintyMap = np.transpose(uncertaintyMap, (2, 1, 0))
        # Read the original MRI DICOM series. 
        image3D_MRI, series_reader_MRI = self.readDicomSeries2SITK(input_directory_with_DICOM_series)
        # Create image data from uncertainty map to a SimpleITK image object
        uncertaintyMap_SITK = sitk.GetImageFromArray(uncertaintyMap) 
        # Define sitk writer
        writer = sitk.ImageFileWriter()
        # Use the study/series/frame of reference information given in the meta-data
        # dictionary and not the automatically generated information from the file IO
        # Information will be overwritter later
        writer.KeepOriginalImageUIDOn()
        # Get time and date
        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")
        # Get unique new study UID using pydicom
        study_uid = pydicom.uid.generate_uid() 

        # For each slice in the original MRI series
        for i in range(image3D_MRI.GetDepth()):
            # Read the image slice from uncertainty map
            image_slice = uncertaintyMap_SITK[:, :, i]
            # Set DICOM tags for each slice
            # Write patient name as original series
            #image_slice.SetMetaData("0010|0010", patient)
            image_slice.SetMetaData("0010|0010", series_reader_MRI.GetMetaData(i,"0010|0010"))
            # Write patient ID as original series
            #image_slice.SetMetaData("0010|0020", patient)
            image_slice.SetMetaData("0010|0020", series_reader_MRI.GetMetaData(i,"0010|0020"))
            # Set Specific Character Set Attribute to the same as the original series
            image_slice.SetMetaData('0008|0005', series_reader_MRI.GetMetaData(i,"0008|0005"))
            # Set modification time
            image_slice.SetMetaData("0008|0031", modification_time)
            # Set modification date
            image_slice.SetMetaData("0008|0021", modification_date)
            # Set modality to PET
            image_slice.SetMetaData("0008|0060", "PT")
            # Set image type
            image_slice.SetMetaData("0008|0008", "ORIGINAL\PRIMARY")
            # Set study instance
            image_slice.SetMetaData("0020|000d", study_uid)
            # Set series instance
            image_slice.SetMetaData("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time)
            # Set series description
            image_slice.SetMetaData("0008|103e", "AI contour uncertainty map")
            # Instance Creation Date
            image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
            # Instance Creation Time
            image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
            # Instance Number
            image_slice.SetMetaData("0020|0013", str(i))
            # Patient weight
            image_slice.SetMetaData("0010|1030", "75")
            # Patient size
            image_slice.SetMetaData("0010|1020", "1.70")
            # Set frame of reference UID
            image_slice.SetMetaData("0020|0052", series_reader_MRI .GetMetaData(i,"0020|0052"))
            # PET specific tags. 
            # Not sure all is needed (or any) but I will keep them for now.
            image_slice.SetMetaData("0054|1002", "EMISSION")
            # Frame Reference Time
            image_slice.SetMetaData("0054|1300", "0")
            # Performed Procedure Step Start Time
            image_slice.SetMetaData("0040|0245", "102450")
            # Radiopharmaceutical Start Time 
            image_slice.SetMetaData("0018|1072", "092500.00")
            # Scan Progression Direction
            image_slice.SetMetaData("0054|0501", "HEAD_TO_FEET")
            # Content Time
            image_slice.SetMetaData("0008|0033", "103117")
            # PET image index as slice index
            image_slice.SetMetaData("0054|1330", str(i))
            # Actual Frame Duration	
            image_slice.SetMetaData("0018|1242", "90000")
            # Decay correction
            image_slice.SetMetaData("0054|1102", "START")
            # Acquisition Start Condition
            image_slice.SetMetaData("0018|0073", "MANU")
            # Acquisition Termination Condition 
            image_slice.SetMetaData("0018|0071", "TIME")
            # Number of slices  
            image_slice.SetMetaData("0054|0081", str(image3D_MRI.GetDepth()))  
            # Series type
            image_slice.SetMetaData("0054|1000", r"STATIC\IMAGE")
            # Set units
            #image_slice.SetMetaData("0054|1001", "BQML") #CNTS, Counts, BQML, Bq/ml
            image_slice.SetMetaData("0054|1001", "CNTS")
            # The units of the pixel values obtained after conversion from the stored pixel values (SV) (Pixel Data (7FE0,0010)) 
            # to pixel value units (U), as defined by Rescale Intercept (0028,1052) and Rescale Slope (0028,1053).
            # Position Reference Indicator
            image_slice.SetMetaData("0020|1040", "SN")
            # Collimator type
            image_slice.SetMetaData("0018|1181", "NONE")
            # Corrected Image   
            image_slice.SetMetaData("0028|0051", r"DECY\ATTN\SCAT\DTIM\RANSNG\DCAL\SLSENS\NORM")
            # Patient Gantry Relationship Code Sequence
            image_slice.SetMetaData("0054|0414", "1")
            # Patient Orientation Code Sequence
            image_slice.SetMetaData("0054|0410", "1")
            # Radiopharmaceutical Information Sequence
            image_slice.SetMetaData("0054|0016", "1")
            # Manufacturer
            image_slice.SetMetaData("0008|0070", "CJG")
            # Acqusition date
            image_slice.SetMetaData("0008|0022", time.strftime("%Y%m%d"))
            # Acqusition time
            image_slice.SetMetaData("0008|0032", time.strftime("%H%M%S"))
            # Actual frame duration
            image_slice.SetMetaData("0018|1242", "0")
            # Smallest Image Pixel Value
            image_slice.SetMetaData("0028|0106", "0")
            # Largest Image Pixel Value
            image_slice.SetMetaData("0028|0107", "32767")
            # Pixel representation
            image_slice.SetMetaData("0028|0103", "1")
            # Radiopharmaceutical
            image_slice.SetMetaData("0018|0031", "FDG -- fluorodeoxyglucose")
            # Radionuclide Half Life	
            image_slice.SetMetaData("0018|1075", "6586.2001953125")
            # Radionuclide Total Dose
            image_slice.SetMetaData("0018|1074", "299000000")
            # Randoms Correction Method
            image_slice.SetMetaData("0054|1100", "SING")
            # Decay Factor
            image_slice.SetMetaData("0054|1321", "1.00474")
            # Scatter Fraction Factor
            image_slice.SetMetaData("0054|1323", "0.317773")
            # Dead Time Factor
            image_slice.SetMetaData("0054|1324", "1.08806")
            # Slice Sensitivity Factor
            image_slice.SetMetaData("0054|1320", "1")
            # Set Rescale Intercept
            image_slice.SetMetaData("0028|1052", "0")
            # Set Rescale Slope
            image_slice.SetMetaData("0028|1053", "1")
            # Set DoseCalibrationFactor to 1 
            #image_slice.SetMetaData("0054|1322", "1")
            image_slice.SetMetaData("0054|1322", "100")
            # Orientation tags
            # Set slice thickness as original slice thickness
            image_slice.SetMetaData("0018|0050", series_reader_MRI .GetMetaData(i, "0018|0050"))
            # Set image position patient
            image_slice.SetMetaData("0020|0032", series_reader_MRI .GetMetaData(i, "0020|0032"))
            # Set image orientation patient
            image_slice.SetMetaData("0020|0037", series_reader_MRI .GetMetaData(i, "0020|0037"))
            # Set patient position
            image_slice.SetMetaData("0018|5100", series_reader_MRI .GetMetaData(i, "0018|5100"))
            # Set slice location
            image_slice.SetMetaData("0020|1041", series_reader_MRI .GetMetaData(i, "0020|1041"))
            # Set pixel spacing (a bit differently than in the example)
            pixelSpacing = series_reader_MRI .GetMetaData(i, "0028|0030")
            # Separate pixel spacings and get float
            pixelSpacing1 = float(pixelSpacing.split('\\')[0])
            pixelSpacing2 = float(pixelSpacing.split('\\')[1])
            # Set spacing 
            image_slice.SetSpacing((pixelSpacing1, pixelSpacing2))
            # Set rows
            image_slice.SetMetaData("0028|0010", series_reader_MRI .GetMetaData(i, "0028|0010"))
            # Set columns
            image_slice.SetMetaData("0028|0011", series_reader_MRI .GetMetaData(i, "0028|0011"))
            # Write to the output directory and add the extension dcm.
            # This force writing in DICOM format.
            # Make sure output folder exist first 
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            # Write data     
            writer.SetFileName(os.path.join(output_directory, str(i) + ".dcm"))
            writer.Execute(image_slice)
