# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Pre-process data for input to nnUNet training. 
# Takes about an hour on a 32 core machine for 100 patients. 
# *********************************************************************************

import os
from joblib import Parallel, delayed
import multiprocessing
import numpy as np 
import matplotlib.pyplot as plt
import shutil

from ioDataMethods import ioDataMethodsClass
from commonConfig import commonConfigClass
from convertDataMethods import convertDataMethodsClass

# Init needed class instances
ioData = ioDataMethodsClass()           # Class for handling and reading data 
conf = commonConfigClass()              # Init config class
convertData = convertDataMethodsClass() # Functions for converting DICOM to Nifti data

# Function called from parallell loop
def patLargeDataLoop(patNr, patient):
    """
    Arg:
        patNr (int): The current patient number
        patient (str): The current patient name            

    Returns:
        Outputs data to directory         
    
    """
    print(patient)
    # Convert data to Nifti and create encoded ground truth segmentation map
    # Convert Dicom to Nifti
    # convertData.DicomRT2Nifti(patNr, patient, conf.preProcess.inputDicomPatientDir, conf.preProcess.outputNiftiPatientDir)
    # Get file list over outputed Nifti structures
    niiStructList = os.listdir(os.path.join(conf.preProcess.outputNiftiPatientDir, patient))
    # Make sure it is only Nifti files
    niiStructList = [x for x in niiStructList if x.endswith(conf.preProcess.fileSuffix)]
    # Create lowercase list of structures
    niiStructList_lowercase = [x.lower() for x in niiStructList]
    # Make sure the image volume is within the list
    assert conf.preProcess.imageVolumeName.lower() in niiStructList_lowercase, 'Image volume not found in Nifti structure list for patient: ' + patient
    # Copy the image volume to the training data output directory
    newImageVolumeNamePath = os.path.join(conf.preProcess.outputTrainingDataDir, conf.preProcess.TrainingDataImageDir , patient + '_0000' + conf.preProcess.fileSuffix)
    os.makedirs(os.path.dirname(newImageVolumeNamePath), exist_ok=True)
    shutil.copy(os.path.join(conf.preProcess.outputNiftiPatientDir, patient, conf.preProcess.imageVolumeName), newImageVolumeNamePath) 

    # Create encoded ground truth segmentation data
    # Check if variable exists, it should not
    try: encodedImageGT
    except NameError: encodedImageGT = None
    # Start loop
    for structure, encodedValue in conf.preProcess.structureOfInterestDict.items():
        # Poke loop counter
        #i_loop += 1
        try: 
            # Get index of structure in from lowercase list
            structureIndex = niiStructList_lowercase.index(conf.preProcess.fileMaskPrefix + structure.lower() + conf.preProcess.fileSuffix)
        except:
            # Structure not found in list
            print('Structure not found in list: ' + structure)
            # Ignore the structure and continue with next structure
            continue
        
        # Get structure path from the files 
        structureFilePath = os.path.join(conf.preProcess.outputNiftiPatientDir, patient, niiStructList[structureIndex])
        # Get structure data
        structureData, sitk_structureData, pxlspacing_structureData = ioData.readNiftiFile(structureFilePath)
        # Assert max value to be 1
        assert np.max(structureData) == 1, 'Max value in structure data must be 1'
        # Assert min value to be 0
        assert np.min(structureData) == 0, 'Min value in structure data must be 0'
        # Encode the segmentation data by multiplying with the encoded value
        # If variable does not exist yet, create it
        if encodedImageGT is None:
            encodedImageGT = structureData * encodedValue
        else: # If defined and existing 
            # Add the data each time in the loop
            encodedImageGT = encodedImageGT + (structureData * encodedValue)
        
        # Assert that there is no overlap between the structures
        # The dictionary is defined with increasing values for the structure of interest
        # The largest value must therefore always be equal to encodedValue
        assert np.max(encodedImageGT) == encodedValue, 'Structure overlap found for structure ' + structure + ' with max value ' + str(np.max(encodedImageGT)) + ' and patient ' + patient
    
    # Removed as not all structures are always available in the data
    # Make sure encodedImageGT contains values for all structures of interest
    for structure, encodedValue in conf.preProcess.structureOfInterestDict.items():
        assert np.sum(np.where(encodedImageGT == encodedValue)) > 0 , 'All structures were not found in encodedImageGT' 

    # Save encoded ground truth segmentation data as a new Nifti file     
    # Define new file path for it
    encodedImageGTFilePath = os.path.join(conf.preProcess.outputTrainingDataDir, conf.preProcess.TrainingDataLabelDir, patient + conf.preProcess.fileSuffix)
    # Write to new Nifti file for the encoded ground truth segmentation data
    ioData.writeNiftiFile(encodedImageGT, sitk_structureData, encodedImageGTFilePath)
    

# ### SCRIPT STARTS HERE ###
# Loop over all patient folders and convert DICOM to Nifti
# List patients
patFolders = os.listdir(conf.preProcess.inputDicomPatientDir)
# Only include folder
patFolders = [x for x in patFolders if os.path.isdir(os.path.join(conf.preProcess.inputDicomPatientDir, x))]
# Make sure the output Nifti directory exists
if not os.path.isdir(conf.preProcess.outputNiftiPatientDir):
    # Create dir
    os.mkdir(conf.preProcess.outputNiftiPatientDir)
nrCPU = multiprocessing.cpu_count()-2
#nrCPU = 1

# Init parallell job
patInfo = Parallel(n_jobs=nrCPU, verbose=10)(delayed(patLargeDataLoop)(patNr, patient) for patNr, patient in enumerate(patFolders))
     
# Print message
print('Pre-processing for training data is complete!')
print('A manual copy of the data must now be done to the nnUnet folder structure')
print('Manually copy data folder imagesTr and labelsTr from the final folder to the nnUNet_raw folder')
print('nnUNet_raw/Task***/imagesTr and nnUNet_raw/Task***/labelsTr')
print('Generate the json file for the dataset manually, see comments in this script')
# Insert pause to allow user to read message
input(' ')
input('Press Enter to continue...')
# Print message
print('This program is done, dont forget to copy the data to the nnUnet folder structure')


# Generate the json file for the dataset manually
# Put the json file in the folder
# It can look like this where encoding must match the preprocessing encoding values. 
"""
{ 
 "channel_names": {  
   "1": "CT"
 }, 
 "labels": { 
   "background": 0,
   "Rectum": 1,
   "Bladder": 2
 }, 
 "numTraining": 100, 
 "file_ending": ".nii.gz"
 }
 """
