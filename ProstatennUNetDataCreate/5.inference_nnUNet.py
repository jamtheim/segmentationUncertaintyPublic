# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Pipeline for preparing data for inference and performing inference
# After inference, the segmentation map results from nnUNet is converted to a new DICOM RT structure.
# *********************************************************************************

import os
import sys
from joblib import Parallel, delayed
import multiprocessing
import subprocess
import numpy as np 
import time 
import matplotlib.pyplot as plt
import shutil
import dicom2nifti
from rt_utils import RTStructBuilder
import psutil
import platform
import nibabel as nib
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset
from matplotlib import colormaps

from ioDataMethods import ioDataMethodsClass
from commonConfig import commonConfigClass
from convertDataMethods import convertDataMethodsClass

# Init needed class instances
ioData = ioDataMethodsClass()           # Class for handling and reading data         
convertData = convertDataMethodsClass() # Functions for converting DICOM to Nifti data

# Check if command-line argument was provided as an argument
if len(sys.argv) == 5:
    # Get the current inference subfolder and task number and chosen CUDA device
    # Example: python 5.inference_nnUNet.py MRI-only_test_data 110 0
    currentInferenceSubFolder = sys.argv[1]
    currentTaskNumber = sys.argv[2]
    chosenCUDADevice = sys.argv[3]
    chosenDataFormat = sys.argv[4]
    print('Inference subfolder:', currentInferenceSubFolder)
    print('Task number:', currentTaskNumber)
    print('Chosen CUDA device:', chosenCUDADevice)
    print('Chosen data format:', chosenDataFormat)

    conf = commonConfigClass(currentInferenceSubFolder, currentTaskNumber) # Init config class
else:
    # Print usage instructions if no argument is provided
    print("Usage: python script.py <currinferenceSubFolder> <taskNumber> <chosenCUDADevice> <chosenDataFormat>")
    print('No argument provided, exiting...')
    #sys.exit()
    # If need to debug, use the following and run script directly
    #conf = commonConfigClass('MRI-only_test_data', 110)  # Init config class
    #chosenDataFormat = 'nifti'

# Exit for debug
#sys.exit()
    
# Options
inferenceMode = True #Set to false for debugging 
# Determine OS used
system_name = platform.system()

# Function for checking running processes
def checkIfProcessRunning(processName):
    '''
    Check if there is any running process that contains the given name processName.
    '''
    #Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


if chosenDataFormat == 'dicom':
    # This will read from the DICOM inference directory
    # Loop over all patient folders and convert DICOM to Nifti
    # List patients
    patFolders = os.listdir(conf.inference.inputDicomPatientDir)
    # print(conf.inference.inputDicomPatientDir)

    # Only include folders
    patFolders = [x for x in patFolders if os.path.isdir(os.path.join(conf.inference.inputDicomPatientDir, x))]

    # Make sure the output Nifti directory exists
    if not os.path.isdir(conf.inference.inputNiftiPatientDir):
        os.mkdir(conf.inference.inputNiftiPatientDir)
    # Loop over patients and convert DICOM to Nifti 

    # Function for parallell processing
    def convertDicom2nifti(patient): 
        dicom2nifti.dicom_series_to_nifti(os.path.join(conf.inference.inputDicomPatientDir, patient, conf.preProcess.CTfolder), os.path.join(conf.inference.inputNiftiPatientDir, patient + '_0000' + conf.preProcess.fileSuffix), reorient_nifti=False) 
        # For this to work dicom2nifti must be modified as TR and TE are empty when exporting from Eclipse. 
        # This is done in the file dicom2nifti\convert_generic.py, line 285 in version 2.4.9.
        # # Set TR and TE if available
        # if Tag(0x0018, 0x0080) in dicom_input[0] and Tag(0x0018, 0x0081) in dicom_input[0]:
        #    pass #Edit by CJG
        #    #common.set_tr_te(nii_image, dicom_input[0].RepetitionTime, dicom_input[0].EchoTime)

    # Determine number of CPU cores dicom2nifti
    if len(patFolders) < multiprocessing.cpu_count()-2:
        nrCPU = len(patFolders) 
        #nrCPU = 1
    else: 
        nrCPU = multiprocessing.cpu_count()-2

    # Excecute DICOM to Nifti conversion in parallel
    Parallel(n_jobs=nrCPU, verbose=10)(delayed(convertDicom2nifti)(patient) for patNr, patient in enumerate(patFolders))


# Do inference in nnUNet, takes all data in the folder and outputs segmentation Nifti maps. 
if inferenceMode == True: 
       
    if system_name == 'Linux': 
        # Make sure only one GPU is used for softmax output for every fold. Not adapted to more GPUs yet but can be done.
        if conf.inference.nrGPU == 1: 
            # Print time when inference started
            print('Inference combined started at: ' + time.strftime("%H:%M:%S", time.localtime()))
            # Using all folds for inference (default)
            if conf.inference.numberOfModels == 10:
                command_gpu0 = 'CUDA_VISIBLE_DEVICES=' + str(chosenCUDADevice) + ' nnUNetv2_predict -d ' + str(conf.inference.nnUNetTaskNumber) + ' -i ' + conf.inference.inputNiftiPatientDir  + '  -o ' + conf.inference.outputNiftiSegPatientDir +' -f  0 1 2 3 4 5 6 7 8 9 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans'  + ' -num_parts ' + str(conf.inference.nrGPU) + ' -part_id ' + str(0) + ' --save_probabilities'
                os.system(command_gpu0)
            if conf.inference.numberOfModels == 5:
                command_gpu0 = 'CUDA_VISIBLE_DEVICES=' + str(chosenCUDADevice) + ' nnUNetv2_predict -d ' + str(conf.inference.nnUNetTaskNumber) + ' -i ' + conf.inference.inputNiftiPatientDir  + '  -o ' + conf.inference.outputNiftiSegPatientDir +' -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans'  + ' -num_parts ' + str(conf.inference.nrGPU) + ' -part_id ' + str(0) + ' --save_probabilities'
                os.system(command_gpu0)
            
            print('Inference finished for combined folds!')
            # Print clock time
            print('Inference finished at: ' + time.strftime("%H:%M:%S", time.localtime()))

            # Print time when inference started
            print(' ')
            print(' ')
            print('Inference foldwise started at: ' + time.strftime("%H:%M:%S", time.localtime()))
            # Loop over folds and do inference for each fold separately to get softmax output for each fold. Save in separate folders.
            for foldNr in range(conf.inference.numberOfModels):
                print('This is fold number: ' + str(foldNr))
                foldOutputDir = os.path.join(conf.inference.outputNiftiSegPatientDir, 'fold_' + str(foldNr))
                command_gpu0_fold = 'CUDA_VISIBLE_DEVICES=' + str(chosenCUDADevice) + ' nnUNetv2_predict -d ' + str(conf.inference.nnUNetTaskNumber) + ' -i ' + conf.inference.inputNiftiPatientDir  + '  -o ' + foldOutputDir + ' -f ' + str(foldNr) + ' -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans'  + ' -num_parts ' + str(conf.inference.nrGPU) + ' -part_id ' + str(0) + ' --save_probabilities'
                os.system(command_gpu0_fold)
                print('Inference finished for fold: ' + str(foldNr))
                # Print clock time
                print('Inference finished at: ' + time.strftime("%H:%M:%S", time.localtime()))
                print(' ')


# Convert softmax probability maps to softmax values in Nifti format.
# This is done for the aggregated data and not the fold wise data.
# Get patients in outputNiftiSegPatientDir
patientNiftiSegFiles = os.listdir(conf.inference.outputNiftiSegPatientDir)
# Make sure it is only npz files
patientNpzFiles = [x for x in patientNiftiSegFiles if x.endswith('.npz')]
# Loop over patients
for patientNpzFile in patientNpzFiles:
    # Get path
    patientNpzFilePath = os.path.join(conf.inference.outputNiftiSegPatientDir, patientNpzFile)
    # Read in softmax array 
    softmaxArray = np.load(patientNpzFilePath)
    # Get softmax array for the probabilities of class 1 (i.e. the structure of interest)
    softmaxArray = softmaxArray['probabilities'][1, :, :, :]
    # Transpose so slices will be in the last dimension
    softmaxArray = np.transpose(softmaxArray, (2,1,0))
    # Get patient ID without file suffix
    patient = patientNpzFile.split('.npz')[0]
    # Set file path
    softmaxFilePath = os.path.join(conf.inference.outputNiftiSegPatientDir, patient + '_softmax.nii.gz')
    # Get the affine matrix
    predLabelaffine = nib.load(os.path.join(conf.inference.outputNiftiSegPatientDir, patient + conf.preProcess.fileSuffix)).affine
    nib.save(nib.Nifti1Image(softmaxArray, predLabelaffine), softmaxFilePath)
    print('Softmax array for patient: ' + patient + ' loaded and converted to Nifti format!')
    # Remove the npz file
    os.remove(patientNpzFilePath)
    print('Npz file for patient: ' + patient + ' removed after conversion!')


# Convert Nifti segmentation maps to RT structures
# Loop over patients in outputNiftiSegPatientDir
patientNiftiSegFiles = os.listdir(conf.inference.outputNiftiSegPatientDir)
# Make sure it is only Nifti files
patientNiftiSegFiles = [x for x in patientNiftiSegFiles if x.endswith(conf.preProcess.fileSuffix)]
# Make sure they do not contain the uncertainty map
patientNiftiSegFiles = [x for x in patientNiftiSegFiles if not x.endswith('uncertaintyMap.nii.gz')]
# Make sure they do not contain the softmax map
patientNiftiSegFiles = [x for x in patientNiftiSegFiles if not x.endswith('softmax.nii.gz')]

print(patientNiftiSegFiles)

# Define function for parallel processing
def convertNifti2RTstruct(file): 
    # Read the Nifti segmentation file
    np_segData, sitk_segData, pixelSpacing = ioData.readNiftiFile(os.path.join(conf.inference.outputNiftiSegPatientDir, file))
    # Get patient name as the file name without the file suffix
    patient = file.split(conf.preProcess.fileSuffix)[0]
    # Get patient original MR folder
    patientCTfolder = os.path.join(conf.inference.inputDicomPatientDir, patient, 'MR_StorT2')
    # Create new RT Struct. Requires the DICOM series path for the RT structure, i.e. the MR folder.
    rtstruct1 = RTStructBuilder.create_new(dicom_series_path=patientCTfolder)
    # Add a second RT struct so we can create a copy of it that is still unique but has the correct DICOM reference
    # This is used in the second round of the observer study when the structure needs to be restored to the original state. 
    rtstruct2 = RTStructBuilder.create_new(dicom_series_path=patientCTfolder)

    # Add 3D masks as ROIs to the RT structure
    # Loop over the structures defined in pre-processing and set custom names and RGB codes. 
    # Reaons why pre-process info is used here is that this is how the model was trained. 
    for structureName, encodedValue in conf.inference.structureOfInterestDict.items():
        # Add ROI to RT structure if structure is not empty
        if np.sum(np_segData == encodedValue) > 0:
            # print message
            print('Adding structure: ' + structureName + ' to RT struct for patient: ' + patient)
            
            rtstruct1.add_roi( 
            mask = np_segData == encodedValue, # Pick correct mask for the structure 
            color = conf.inference.inferenceRGBcodeDict[structureName], # Get color from config file
            name = conf.inference.inferenceNameDict[structureName], # Get name from config file
            approximate_contours = False, # Higher precision but larger file
            roi_generation_algorithm = 0, # Automatic ROI generation
            )

            rtstruct2.add_roi( 
            mask = np_segData == encodedValue, # Pick correct mask for the structure 
            color = conf.inference.inferenceRGBcodeDict[structureName], # Get color from config file
            name = conf.inference.inferenceNameDict[structureName], # Get name from config file
            approximate_contours = False, # Higher precision but larger file
            roi_generation_algorithm = 0, # Automatic ROI generation
            )
        
    # If segmentation was not totally empty save RT struct file to output folder
    if np.sum(np_segData) > 0:
        # Set new file path for RT struct DICOM file
        rtstruct1FilePath = os.path.join(conf.inference.outputRTstructPatientDir, patient + '_' + str(conf.inference.nnUNetTaskNumber) + '_RTstruct.dcm')
        rtstruct2FilePath = os.path.join(conf.inference.outputRTstructPatientDir, patient + '_' + str(conf.inference.nnUNetTaskNumber) + '_RTstruct_secondRound.dcm')
        # Save the RT structures to file as a DICOM RT structure 
        rtstruct1.save(rtstruct1FilePath)
        rtstruct2.save(rtstruct2FilePath)
    else: 
        print('No segmentation data found for patient: ' + patient)


# Determine number of CPU cores convertNifti2RTstruct
if len(patientNiftiSegFiles) < multiprocessing.cpu_count()-2:
    nrCPU_RTstruct = len(patientNiftiSegFiles) 
else: 
    nrCPU_RTstruct = multiprocessing.cpu_count()-2
        
# Excecute RT struct generation in parallel
Parallel(n_jobs=nrCPU_RTstruct, verbose=10)(delayed(convertNifti2RTstruct)(file) for fileNr, file in enumerate(patientNiftiSegFiles))


# Assert that all files have been created, else give error. Removed as empty files are created for empty segmentations but no RT structs 
# created for those patients.
#outputRTstructFileList = os.listdir(conf.inference.outputRTstructPatientDir)
#outputRTstructFileList = [x for x in outputRTstructFileList if x.endswith('.dcm')]
#assert len(patientNiftiSegFiles) == len(patFolders), 'Not all output segmentation files have been created'
#assert len(outputRTstructFileList) == len(patFolders), 'Not all output RT DICOM files have been created'
#assert len(outputRTstructFileList) == len(patientNiftiSegFiles), 'Not all output RT DICOM files have been created'
#print(' ')

# Print message
print('Inference and RT struct generation is complete!')
print('Please check the output folder for the RT structures!')
print(' ')


# Calculate uncertainty map as variations in softmax output between folds
# This also outputs it to a DICOM series folder. This can be used to display uncertainty 
# as a PET image in Varian Eclipse. 
# Loop over patients in outputNiftiSegPatientDir
for patientFile in patientNiftiSegFiles: 
    # Get patient name as the file name without the file suffix
    patient = patientFile.split(conf.preProcess.fileSuffix)[0]
    # Get uncertainty map file path
    uncertaintyMapFilePath = os.path.join(conf.inference.outputNiftiSegPatientDir, patient + '_uncertaintyMap.nii.gz')
    
    # If it does not exist calculate it
    if not os.path.isfile(uncertaintyMapFilePath):
        print('Calculating uncertainty map for patient: ' + patient)
        # Read the Nifti segmentation file
        predLabel = nib.load(os.path.join(conf.inference.outputNiftiSegPatientDir, patientFile)).get_fdata().astype(np.float32)
        # Get the affine matrix
        predLabelaffine = nib.load(os.path.join(conf.inference.outputNiftiSegPatientDir, patientFile)).affine
        # Read in softmax array for each fold and calculate uncertainty map
        # Loop over folds
        for foldNr in range(conf.inference.numberOfModels): 
            print('Reading softmax array for fold: ' + str(foldNr) + ' for patient: ' + patient)
            # Get softmax values for each fold
            foldOutputDir = os.path.join(conf.inference.outputNiftiSegPatientDir, 'fold_' + str(foldNr))
            softmaxFoldFilePath = os.path.join(foldOutputDir, patient + '.npz')
            # Read the numpy array and save to array
            softmaxArray = np.load(softmaxFoldFilePath)
            # Get softmax array for the probabilities of class 1 (i.e. the structure of interest)
            softmaxArray = softmaxArray['probabilities'][1, :, :, :]
            # Put softmax array in numpy array
            if foldNr == 0: 
                softmaxAllFolds = softmaxArray[np.newaxis, ...]
            else:
                softmaxAllFolds = np.concatenate((softmaxAllFolds, softmaxArray[np.newaxis, ...]), axis=0)
            # Remove the npz file for distortion experiments and observer data (saves a lot of space as they are not needed anymore)
            if 'dist' in softmaxFoldFilePath or 'obs' in softmaxFoldFilePath:
                os.remove(softmaxFoldFilePath)
            
        # Calculate uncertainty map for fold
        uncertaintyMap = np.std(softmaxAllFolds, axis=0)
        # Transpose so slices will be in the last dimension
        uncertaintyMap = np.transpose(uncertaintyMap, (2,1,0))
        # Save the uncertainty map as a Nifti file (do not multply it with the segmentation, want to have uncertainty map for all voxels) 
        nib.save(nib.Nifti1Image(uncertaintyMap, predLabelaffine), uncertaintyMapFilePath)
        # predLabel
        print('Uncertainty map for patient: ' + patient + ' saved to Nifti file!')
        print(' ')


    # Convert Nifti uncertainty map to DICOM and output in folder 
    # Set folder paths
    DicomDataIn = os.path.join(conf.inference.inputDicomPatientDir, patient, conf.preProcess.CTfolder)
    uncertaintyDicomDataOut = os.path.join(conf.inference.outputNiftiSegPatientDir, str(patient) + '_' + str(conf.inference.nnUNetTaskNumber) + '_uncertaintyMap_DICOM')
    # Run conversion function
    convertData.uncertaintyNifti2DICOM(patient, DicomDataIn, uncertaintyDicomDataOut, uncertaintyMapFilePath)

    # Copy prostate or rectum data to a structure folder format that is compatible with the task of importing it to Varian Eclipse
    # Copy the folder with MRI DICOM to conf.inference.exportEclipseDir
    shutil.copytree(os.path.join(conf.inference.inputDicomPatientDir, patient, 'MR_StorT2'), os.path.join(conf.inference.exportEclipseDir, patient, 'Step1', 'MR_StorT2'), dirs_exist_ok=True)
    # Create dir for step 2
    os.makedirs(os.path.join(conf.inference.exportEclipseDir, patient, 'Step2'), exist_ok=True)
    # Copy the generated DICOM RT structure to conf.inference.exportEclipseDir
    shutil.copy(os.path.join(conf.inference.outputRTstructPatientDir, patient + '_' + str(conf.inference.nnUNetTaskNumber) + '_RTstruct.dcm'), os.path.join(conf.inference.exportEclipseDir, patient, 'Step1', patient + '_' + str(conf.inference.nnUNetTaskNumber) + '_RTstruct.dcm'))
    shutil.copy(os.path.join(conf.inference.outputRTstructPatientDir, patient + '_' + str(conf.inference.nnUNetTaskNumber) + '_RTstruct_secondRound.dcm'), os.path.join(conf.inference.exportEclipseDir, patient, 'Step2', patient + '_' + str(conf.inference.nnUNetTaskNumber) + '_RTstruct_secondRound.dcm'))
    # Copy the folder uncertaintyDicomDataOut to conf.inference.exportEclipseDir
    shutil.copytree(uncertaintyDicomDataOut, os.path.join(conf.inference.exportEclipseDir, patient, 'Step2', patient + '_' + str(conf.inference.nnUNetTaskNumber) + '_uncertaintyMap_DICOM'), dirs_exist_ok=True)
    
 