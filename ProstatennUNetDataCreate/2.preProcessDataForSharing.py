# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Convert data from DICOM to Nifti to share on public repository.
# *********************************************************************************

import os
from joblib import Parallel, delayed
import multiprocessing
import numpy as np 
import matplotlib.pyplot as plt
import shutil
import json
import random

from ioDataMethods import ioDataMethodsClass
from commonConfig import commonConfigClass
from convertDataMethods import convertDataMethodsClass

# Init needed class instances
ioData = ioDataMethodsClass()           # Class for handling and reading data 
conf = commonConfigClass()              # Init config class
convertData = convertDataMethodsClass() # Functions for converting DICOM to Nifti data

# Function called from parallell loop
def patImageDataLoop(patNr, patient, inputDir, outputDir, imageFolder, contourFolder):
    """
    Arg:
        patNr (int): The current patient number
        patient (str): The current patient name    
        inputDir (str): The input directory with the DICOM data
        outputDir (str): The output directory for the Nifti data
        imageFolder (str): The folder with the image data
        contourFolder (str): The folder with the contour data        

    Returns:
        Outputs data to directory         
    
    """
    print(patient)
    # Convert Dicom images to Nifti
    convertData.DicomRT2NiftiForPublic(patNr, patient, inputDir, outputDir, imageFolder, contourFolder)


def patImageReg2MRILoop(patNr, patient, outputDir, sCTFolder, MRIFolder):
    """
    Arg:
        patNr (int): The current patient number
        patient (str): The current patient name          
        inputDir (str): The input directory with the DICOM data
        outputDir (str): The output directory for the Nifti data
        sCTFolder (str): The folder with the sCT data
        MRIFolder (str): The folder with the MRI data
    Returns:
        Outputs data to directory         
    
    """
    print("Registering sCT to MRI for patient: ", patient)
    # Register sCT to MRI coordinates and voxel size
    sCTMatrixSizeOrig, sCTVoxelSizeOrig, mriMatrixSize, mriVoxelSize, doseMatrixSizeOrig, doseVoxelSizeOrig, doseMin, doseMax = convertData.regsCT2MRIForPublic(patNr, patient, outputDir, sCTFolder, MRIFolder)
    # Return 
    return patient, sCTMatrixSizeOrig, sCTVoxelSizeOrig, mriMatrixSize, mriVoxelSize, doseMatrixSizeOrig, doseVoxelSizeOrig, doseMin, doseMax


def patDoseDataLoop(patNr, patient, inputDir, outputDir, imageFolder, doseDataFolder):
    """
    Arg:
        patNr (int): The current patient number
        patient (str): The current patient name          
        inputDir (str): The input directory with the DICOM data
        outputDir (str): The output directory for the Nifti data
        doseDataFolder (str): The folder with the dose data  
        imageFolder (str): The folder with the image data

    Returns:
        Outputs data to directory         
    
    """
    print(patient)
    # Convert Dicom dose image to Nifti
    convertData.DicomDose2Nifti(patNr, patient, inputDir, outputDir, imageFolder, doseDataFolder)


def patFiducialMRIDataLoop(patNr, patient, inputDir, inputDirNifti, outputDir, imageFolder, contourFolder):
    """
    Arg:
        patNr (int): The current patient number
        patient (str): The current patient name          
        inputDir (str): The input directory with the DICOM data
        outputDir (str): The output directory for the Nifti data
        imageFolder (str): The folder with the image data
        fiducialDataFolder (str): The folder with the RT fiducial struct data

    Returns:
        Outputs data to directory         
    
    """
    print(patient)
    # Convert Dicom RT fiducial struct to Nifti and text file
    convertData.DicomFiducial2Nifti(patNr, patient, inputDir, inputDirNifti, outputDir, imageFolder, contourFolder)


def patObsDataLoop(patNr, patient, outputDir):
    """
    Arg:
        patNr (int): The current patient number
        patient (str): The current patient name          
        outputDir (str): The output directory for the Nifti data

    Returns:
        Outputs data to directory         
    
    """
    # Copy observer data from the uncertainty study to the output directory and rename the files
    # Define file to anon list
    csv_file_path_anonList = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/patient_ID_mappings.csv')
    print(patient)
    # Patients are pseudo anonomized in the uncertainty study so wee need to get the new ID
    anonPatient = convertData.getStudyAnonSubjectName(patient, csv_file_path_anonList)
    # Print the new ID
    print(anonPatient)
    # Define root directory for observer data
    obsRootDir = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/inference/observerDataEdited/')
    # Loop over observers
    for observer in ['obsB', 'obsC', 'obsD', 'obsE']: 
        # Create directory path for observer
        obsDir = os.path.join(obsRootDir, observer)
        # Define the patient directory
        obsPatDir = os.path.join(obsDir, anonPatient + '_' + observer)
        # Define paths for each structure of interest
        # Prostate
        prostateStep1Path = os.path.join(obsPatDir, 'step1Edited', 'CTV1', 'Nifti', 'mask_' + observer + '_' + anonPatient + '_CTV_step1.nii.gz')
        prostateStep2Path = os.path.join(obsPatDir, 'step2Edited', 'CTV2', 'Nifti', 'mask_' + observer + '_' + anonPatient + '_CTV_step2.nii.gz')
        rectumStep1Path = os.path.join(obsPatDir, 'step1Edited', 'Rectum1', 'Nifti', 'mask_' + observer + '_' + anonPatient + '_Rectum_step1.nii.gz')
        rectumStep2Path = os.path.join(obsPatDir, 'step2Edited', 'Rectum2', 'Nifti', 'mask_' + observer + '_' + anonPatient + '_Rectum_step2.nii.gz')
        # Copy the observer data to the output directory and rename the files
        # Define directory to copy data to
        prostateDirCopy = os.path.join(outputDir, patient, 'MR_StorT2', 'observerData')
        # Make sure the output directory exists
        os.makedirs(prostateDirCopy, exist_ok=True)
        
        # Prostate
        # Copy the prostate step 1 structure
        shutil.copyfile(os.path.join(prostateStep1Path), os.path.join(prostateDirCopy, 'mask_CTVT_427_step1_' + observer + '.nii.gz'))
        # Copy the prostate step 2 structure
        shutil.copyfile(os.path.join(prostateStep2Path), os.path.join(prostateDirCopy, 'mask_CTVT_427_step2_' + observer + '.nii.gz'))
        
        # Rectum
        # Copy the rectum step 1 structure
        shutil.copyfile(os.path.join(rectumStep1Path), os.path.join(prostateDirCopy, 'mask_Rectum_step1_' + observer + '.nii.gz'))
        # Copy the rectum step 2 structure
        shutil.copyfile(os.path.join(rectumStep2Path), os.path.join(prostateDirCopy, 'mask_Rectum_step2_' + observer + '.nii.gz'))

      

def patnnUNetDataLoop(patNr, patient, outputDir):
    """
    Arg:
        patNr (int): The current patient number
        patient (str): The current patient name          
        outputDir (str): The output directory for the Nifti data

    Returns:
        Outputs data to directory         
    
    """
    print(patient)
    # Define prostate and rectum nnUNet output directories
    prostate_nnUNetDir = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/inference/MRI-only_test_data/inference_NiftiSeg_out_TaskNumber_110/')
    rectum_nnUNetDir = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/inference/MRI-only_test_data/inference_NiftiSeg_out_TaskNumber_111/')
    # Copy data from nnUNet output to output directory

    # Prostate
    # Define directory to copy data to
    prostateDirCopy = os.path.join(outputDir, patient, 'MR_StorT2', 'nnUNetOutput')
    # Make sure the output directory exists
    os.makedirs(prostateDirCopy, exist_ok=True)
    # Copy the structure file
    shutil.copyfile(os.path.join(prostate_nnUNetDir, patient + '.nii.gz'), os.path.join(prostateDirCopy, 'mask_CTVT_427_nnUNet.nii.gz'))
    # Copy the uncertainty map
    shutil.copyfile(os.path.join(prostate_nnUNetDir, patient + '_uncertaintyMap.nii.gz'), os.path.join(prostateDirCopy, 'mask_CTVT_427_nnUNet_uncertaintyMap.nii.gz'))
    # Make sure fold folder exists
    os.makedirs(os.path.join(prostateDirCopy, 'folds'), exist_ok=True)
    # Copy the output structure from each fold, fold0 to fold9
    for foldNr in range(10):
        shutil.copyfile(os.path.join(prostate_nnUNetDir, 'fold_' + str(foldNr), patient + '.nii.gz'), os.path.join(prostateDirCopy, 'folds', 'mask_CTVT_427_nnUNet_fold_' + str(foldNr) + '.nii.gz'))

    # Rectum
    # Define directory to copy data to
    rectumDirCopy = os.path.join(outputDir, patient, 'MR_StorT2', 'nnUNetOutput')
    # Make sure the output directory exists
    os.makedirs(rectumDirCopy, exist_ok=True)
    # Copy the structure file
    shutil.copyfile(os.path.join(rectum_nnUNetDir, patient + '.nii.gz'), os.path.join(rectumDirCopy, 'mask_Rectum_nnUNet.nii.gz'))
    # Copy the uncertainty map
    shutil.copyfile(os.path.join(rectum_nnUNetDir, patient + '_uncertaintyMap.nii.gz'), os.path.join(rectumDirCopy, 'mask_Rectum_nnUNet_uncertaintyMap.nii.gz'))
    # Make sure fold folder exists
    os.makedirs(os.path.join(prostateDirCopy, 'folds'), exist_ok=True)
    # Copy the output structure from each fold, fold0 to fold9
    for foldNr in range(10):
        shutil.copyfile(os.path.join(rectum_nnUNetDir, 'fold_' + str(foldNr), patient + '.nii.gz'), os.path.join(rectumDirCopy, 'folds', 'mask_Rectum_nnUNet_fold_' + str(foldNr) + '.nii.gz'))

def patQADataLoop(patNr, patient, outputDir):
    """
    Arg:
        patNr (int): The current patient number
        patient (str): The current patient name          
        outputDir (str): The output directory for the Nifti data

    Returns:
        Outputs data to directory         
    
    """
    # Define a dictionary with missing data for each patient

    # Function to load the dictionary from file
    def load_missing_data(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        # Convert list to set for each value
        return {key: set(value) for key, value in data.items()}

    # Load the missing data dictionary
    file_path = '/mnt/mdstore2/Christian/MRIOnlyData/missingStructuresAll.json'
    expectedMissingDataDict = load_missing_data(file_path)

    # QA data for each patient directory to make sure it is correct
    # Create patient folder path and call QA function
    patientDirMRI = os.path.join(outputDir, patient, 'MR_StorT2')
    convertData.qaPublicData(patientDirMRI, patient, 1, expectedMissingDataDict)
    # sCT
    patientDirsCT = os.path.join(outputDir, patient, 'sCT')
    convertData.qaPublicData(patientDirsCT, patient, 0, expectedMissingDataDict)
    

def writeGeometryAndDoseResults(outputPatientDir, geomAndDoseResults):
    """
    Write out the geometry results to a csv file
    """
    dataType = None
    if 'shareData_MRI-only from DATE' in outputPatientDir:
        dataType = 'Training data'
        geomResultsCSVPath = '/mnt/mdstore2/Christian/MRIOnlyData/patGeometryInformation_TrainingData.csv'
    if 'shareData_MRI-only DATERANGE test data' in outputPatientDir:
        dataType = 'Test data'
        geomResultsCSVPath = '/mnt/mdstore2/Christian/MRIOnlyData/patGeometryInformation_TestData.csv'
    # Write out the results to a csv file
    with open(geomResultsCSVPath, 'w') as file:
        file.write('patient;sCTMatrixSizeOrig;sCTVoxelSizeOrig (mm);mriMatrixSize;mriVoxelSize (mm);doseMatrixSizeOrig;doseVoxelSizeOrig;doseMin (Gy);doseMax (Gy) \n')
        for result in geomAndDoseResults:
            file.write(result[0] + ';' + str(result[1]) + ';' + str(result[2]) + ';' + str(result[3]) + ';' + str(result[4]) + ';' + str(result[5]) + ';' + str(result[6]) + ';' + str(result[7]) + ';' + str(result[8]) +  '\n')
    

def anonData(outputPatientDir, patFolders):
    """
    Args: 
        outputPatientDir (str): The output directory for the Nifti data
        patFolders (list): List of patient folders
    """

    # Define a function which reads a text file and replaces the word given as input argument
    # with another word given as second input argument. There might be multiple occurences of the word.
    def replaceWordInFile(file_path, old_word, new_word):
        """
        Replace a word in a text file
        """
        with open(file_path, 'r') as file:
            filedata = file.read()
        # Replace the target string
        filedata = filedata.replace(old_word, new_word)
        # Write the file out again
        with open(file_path, 'w') as file:
            file.write(filedata)
    
    
    def generateAnonHexID(length):
        """
        Generate a random hex ID with a given length
        """
        # Generate a random hex ID
        return ''.join(random.choice('0123456789abcdef') for i in range(length))


    # Determine if this is training or test data 
    dataType = None
    if 'shareData_MRI-only from DATE' in outputPatientDir:
        dataType = 'Training data'
        structuresMoreThan1ConCompPath = '/mnt/mdstore2/Christian/MRIOnlyData/StructuresMoreThan1ConComp_TrainingData.txt'
        missingStructuresPath = '/mnt/mdstore2/Christian/MRIOnlyData/missingStructures_TrainingData.json'
        patGeometryInformationPath = '/mnt/mdstore2/Christian/MRIOnlyData/patGeometryInformation_TrainingData.csv'
    if 'shareData_MRI-only DATERANGE test data' in outputPatientDir:
        dataType = 'Test data'
        structuresMoreThan1ConCompPath = '/mnt/mdstore2/Christian/MRIOnlyData/StructuresMoreThan1ConComp_TestData.txt'
        missingStructuresPath = '/mnt/mdstore2/Christian/MRIOnlyData/missingStructures_TestData.json'
        patGeometryInformationPath = '/mnt/mdstore2/Christian/MRIOnlyData/patGeometryInformation_TestData.csv'

    # Create a copy of the StructresMoreThan1ConComp file on disk with prefix anon
    # Set path
    structuresMoreThan1ConCompPathAnon = structuresMoreThan1ConCompPath.replace('.txt', '_anon.txt')
    shutil.copyfile(structuresMoreThan1ConCompPath, structuresMoreThan1ConCompPathAnon)    
    # Do the same for missingStructures.json
    missingStructuresPathAnon = missingStructuresPath.replace('.json', '_anon.json')
    shutil.copyfile(missingStructuresPath, missingStructuresPathAnon)
    # Do the same for the geometry information
    patGeometryInformationPathAnon = patGeometryInformationPath.replace('.csv', '_anon.csv')
    shutil.copyfile(patGeometryInformationPath, patGeometryInformationPathAnon)
    
 
    # Loop over all patients and change the folder name       
    for patNr, patient in enumerate(patFolders):
        # extract oldAcq or newAcq
        acq = patient.split('_')[0]
        # Define new ID for patient
        newID = generateAnonHexID(16)
        # Prefix with oldAcq or newAcq
        newID = acq + '_' + newID
        # Print the old and new ID
        print ('Old ID: ', patient, ' New ID: ', newID)
        # Make sure new folder name does not exist
        assert not os.path.isdir(os.path.join(outputPatientDir, newID)), 'Folder already exists'
        # Change the folder name
        os.rename(os.path.join(outputPatientDir, patient), os.path.join(outputPatientDir, newID))
        # Assert existing after change
        assert os.path.isdir(os.path.join(outputPatientDir, newID)), 'Folder does not exist'
        # Change the name in the structuresMoreThan1ConComp anon file
        replaceWordInFile(structuresMoreThan1ConCompPathAnon, patient, newID)
        # Change the name in the missingStructures anon file
        replaceWordInFile(missingStructuresPathAnon, patient, newID)
        # Change the name in the geometry information anon file
        replaceWordInFile(patGeometryInformationPathAnon, patient, newID)
        
 

#### SCRIPT STARTS HERE ###
def processFiles(inputPatientDir, inputPatientDirNifti, outputPatientDir, DataType): 
    """
    Convert the data from DICOM to Nifti format.
    Args: 
        inputPatientDir (str): The input directory with the DICOM data
        outputPatientDir (str): The output directory for the Nifti data
    """
    # nrCPU = multiprocessing.cpu_count()-2
    nrCPU = 60
    
    # Loop over all patient folders and convert DICOM to Nifti
    # List patients, only include folder
    patFolders = os.listdir(inputPatientDir)
    patFolders = [x for x in patFolders if os.path.isdir(os.path.join(inputPatientDir, x))]

    # Remove these patients which has only two fiducial markers
    # I do this so they cant be indentified in the data
    if 'oldAcq_XXX1' in patFolders:
        patFolders.remove('oldAcq_YYY')
    if 'oldAcq_XXX2' in patFolders:
        patFolders.remove('oldAcq_ZZZ')

    # Take only the first patient for testing
    #patFolders = patFolders[0:2] ####
    # Make sure the output Nifti directory exists
    if not os.path.isdir(outputPatientDir):
        os.mkdir(outputPatientDir)
    # Init parallell job
    # sCT, takes a long time
    #Parallel(n_jobs=nrCPU, verbose=10)(delayed(patImageDataLoop)(patNr, patient, inputPatientDir, outputPatientDir, 'sCT', 'sCTContours') for patNr, patient in enumerate(patFolders))
    # MRI, takes a long time
    #Parallel(n_jobs=nrCPU, verbose=10)(delayed(patImageDataLoop)(patNr, patient, inputPatientDir, outputPatientDir, 'MR_StorT2', 'sCTContours') for patNr, patient in enumerate(patFolders))
    # Dose interpolated and original for sCT
    Parallel(n_jobs=nrCPU, verbose=10)(delayed(patDoseDataLoop)(patNr, patient, inputPatientDir, outputPatientDir, 'sCT', 'DoseDist') for patNr, patient in enumerate(patFolders))
    # Dose interpolated for MR
    Parallel(n_jobs=nrCPU, verbose=10)(delayed(patDoseDataLoop)(patNr, patient, inputPatientDir, outputPatientDir, 'MR_StorT2', 'DoseDist') for patNr, patient in enumerate(patFolders))
    # Register sCT to MRI coordinates and voxel size. Also get dose information 
    geomAndDoseResults = Parallel(n_jobs=nrCPU, verbose=10)(delayed(patImageReg2MRILoop)(patNr, patient, outputPatientDir, 'sCT', 'MR_StorT2',) for patNr, patient in enumerate(patFolders))
    # Write out results to a csv file using headlines patient, sCTMatrixSizeOrig, sCTVoxelSizeOrig, mriMatrixSize, mriVoxelSize, doseMatrixSizeOrig, doseVoxelSizeOrig, doseMin, doseMax
    # Must be placed after dose is converted to Nifti so we have access to the the dose information
    writeGeometryAndDoseResults(outputPatientDir, geomAndDoseResults)
    # MR T2 fiducial
    Parallel(n_jobs=nrCPU, verbose=10)(delayed(patFiducialMRIDataLoop)(patNr, patient, inputPatientDir, inputPatientDirNifti, outputPatientDir, 'MR_StorT2', 'fiducialStorT2') for patNr, patient in enumerate(patFolders))
    # If DataType is inferenceData (i.e do for inference data only)
    if DataType == 'inferenceData':
        # Copy nnUNet output and uncertainty map to output directory
        Parallel(n_jobs=nrCPU, verbose=10)(delayed(patnnUNetDataLoop)(patNr, patient, outputPatientDir) for patNr, patient in enumerate(patFolders))
        # Copy observer data from the uncertainty study to the output directory and rename the files
        Parallel(n_jobs=nrCPU, verbose=10)(delayed(patObsDataLoop)(patNr, patient, outputPatientDir) for patNr, patient in enumerate(patFolders))
    # QA the data to make sure it is correct
    nrCPU = 4 # Set to limit GPU memory usage for binary mask object label function
    Parallel(n_jobs=nrCPU, verbose=10)(delayed(patQADataLoop)(patNr, patient, outputPatientDir) for patNr, patient in enumerate(patFolders))
    # Rename the patient folders to the new ID. New id is also propagated to the list describing the missing data and 
    # structures that has more than 1 connected component. Also geometry information. 
    anonData(outputPatientDir, patFolders)

        


print('Processing training data')
# Training data 
inputPatientDirTraining = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/MRI-only from DATE Cleaned and Renamed/')
inputPatientDirTrainingNifti = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/MRI-only from DATE Cleaned and Renamed Nifti/')
outputPatientDirTraining = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/shareData_MRI-only from DATE/')
processFiles(inputPatientDirTraining, inputPatientDirTrainingNifti, outputPatientDirTraining, 'trainingData')

print('Processing test data')
# Test data 
inputPatientDirTest = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/MRI-only DATERANGE test data Cleaned and Renamed/')
inputPatientDirTestNifti = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/MRI-only DATERANGE test data Cleaned and Renamed Nifti/') # Change
outputPatientDirTest = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/shareData_MRI-only DATERANGE test data/')
processFiles(inputPatientDirTest, inputPatientDirTestNifti, outputPatientDirTest, 'inferenceData')

# Print message
print('DICOM has been converted to Nifti!')
