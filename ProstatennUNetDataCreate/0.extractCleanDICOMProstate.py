# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: This script was used to anonomize the prostate MRI only data that was extracted from the 
# clinical TPS database. The data was then used to train the segmentation model.
# *********************************************************************************
import os
import numpy as np
import os.path
import shutil
import fnmatch
from joblib import Parallel, delayed
import multiprocessing
import SimpleITK as sitk
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
import pydicom as dicom
import csv
from collections import Counter


# Support functions for checking data quality
def count_files_in_dir(dir_path):
    """
    Count number of files in a directory and return their paths
    dir_path: Path to directory
    """
    total = 0
    file_paths = []
    for root, dirs, files in os.walk(dir_path):
        total += len(files)
        for file in files:
            file_paths.append(os.path.join(root, file))
    return total, file_paths


def count_dirs_in_dir(dir_path):
    """
    Count number of directories in a directory
    dir_path: Path to directory
    """
    total = 0
    for root, dirs, files in os.walk(dir_path):
        total += len(dirs)
    return total


def find_folder(dir_path, search_word):
    """
    Search for a sub folder in a directory that contains a specific word
    dir_path: Path to directory
    search_word: Word to search for in folder name
    """
    found_folders = []
    for root, dirs, files in os.walk(dir_path):
        for dir in dirs:
            if search_word in dir:
                found_folders.append(os.path.join(root, dir))
    # Found folder must not be empty and should only contain one match
    assert len(found_folders) != 0, "Could not find folder containing the search word  " + search_word + " in " + dir_path
    assert len(found_folders) == 1, "There should be only one folder containing the search word " + search_word + " in " + dir_path
    return found_folders[0]


def find(pattern, path):
    """
    Find a file from a pattern of its name and path 
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    # Assert only one file to exist and found 
    assert len(result) == 1
    # Return data 
    return result


def getSeriesInstanceUID(folder):
    """
    Get series instance UID for the first DICOM file in a folder
    We assume that all DICOM files in the folder are from the same series
    """
    filesOfInterst = os.listdir(folder)
    # Make sure they are DICOM files using list comprehension
    filesOfInterst = [x for x in filesOfInterst if x.endswith('.dcm')]
    # Read the first file in the list using DICOM read
    ds = dicom.read_file(os.path.join(folder, filesOfInterst[0]))
    # Get series instance UID
    seriesInstanceUID = ds.SeriesInstanceUID
    # Return series instance UID
    return seriesInstanceUID


def checkImageVolume(folder): 
    """
    Check that all DICOM slices are included by looking at the slice position and slice thickness
    Check for deviations in slice thickness
    """
     # List to store the dataset
    dataset = []
    # Read all DICOM files in the directory
    for filename in os.listdir(folder):
        if filename.endswith(".dcm"):
            dataset.append(dicom.dcmread(os.path.join(folder, filename)))
    # Sort datasets by Image Position (Patient)
    dataset.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    # Check slice thickness
    slice_thickness = abs(dataset[0].ImagePositionPatient[2] - dataset[1].ImagePositionPatient[2])
    # Loop over all slices and check that the slice thickness is consistent
    for i in range(len(dataset) - 1):
        delta = abs(dataset[i].ImagePositionPatient[2] - dataset[i + 1].ImagePositionPatient[2])
        # Round to 1 decimal
        if round(delta, 1) != round(slice_thickness, 1): 
            print("Slice thickness is inconsistent between slice %d and slice %d" % (i, i + 1))
            print(folder)
            return False
        
    # Check slice order
    for i in range(len(dataset) - 1):
        if dataset[i].ImagePositionPatient[2] > dataset[i + 1].ImagePositionPatient[2]:
            print("Slice order is incorrect between slice %d and slice %d" % (i, i + 1))
            print(folder)
            return False
        
    # If we get here, all slices are included and the slice thickness is consistent
    return True

              
def checkMRAcqProtocol(folder): 
    """
    Check what MR protocol version that has been used for the MR data. 
    The clinic changed to a new acqusition protocol XXX where phase encoding was set to left right 
    instead of anterior posterior. This and other changes were made to improve the image quality.
    Images where reconstructed with AirReconDl by default.  
    Before that we will have a mixture of the old protocol with and without AirReconDl. 
    However, only a small fraction of the patients withh have AirReconDL with the old protocol 
    as XXX broke our protocol and we where forced to create an improved one. 
    """
    # Get the first slice in the folder
    filesOfInterst = os.listdir(folder)
    # Make sure they are DICOM files using list comprehension
    filesOfInterst = [x for x in filesOfInterst if x.endswith('.dcm')]
    # Read the first file in the list using DICOM read
    ds = dicom.read_file(os.path.join(folder, filesOfInterst[0]))
    # We changed to phase encoding left right XXX
    if ds.AcquisitionDate < 'XXX': # Censored
        imageProtocol = 'oldAcq'
    if ds.AcquisitionDate >= 'XXX': # Censored 
        imageProtocol = 'newAcq' # AirReconDL is used by default
    return imageProtocol
    

def getFrameOfReferenceUID(folder):
    """
    Get frame of reference UID for the first DICOM file in a folder
    We assume that all DICOM files in the folder are from the same series
    """
    filesOfInterst = os.listdir(folder)
    # Make sure they are DICOM files using list comprehension
    filesOfInterst = [x for x in filesOfInterst if x.endswith('.dcm')]
    # Read the first file in the list using DICOM read
    ds = dicom.read_file(os.path.join(folder, filesOfInterst[0]))
    # Get series instance UID
    seriesFrameOfReference = ds.FrameOfReferenceUID
    # Return series instance UID
    return seriesFrameOfReference


def referencedSeriesInstanceUID(folder):
    """
    Get referenced series instance UID for a RT struct file in a folder
    """
    # Find the RS struct file path
    RSstructFilePath = find('*.dcm' , os.path.join(folder))
    RSstructFilePath = RSstructFilePath[0] # We demand there to be only one in the find function
    # Read the RS struct file 
    ds = dicom.read_file(RSstructFilePath)
    # Get referencedSeriesInstanceUID
    referencedSeriesInstanceUID = ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
    # Return series instance UID
    return referencedSeriesInstanceUID


### CONFIG ###
# Training data
# Set path to raw data folder
#rawDataInputPath = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/MRI-only XXX training data') 
# Set path to output folder with cleaned data
#outputPathCleaned = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/MRI-only from XXX Cleaned')
# Set path to output folder with renamed data
#outputPathRenamed = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/MRI-only from XXX Cleaned and Renamed')

# Inference data
# Set path to raw data folder
rawDataInputPath = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/MRI-only XXX test data') 
# Set path to output folder with cleaned data
outputPathCleaned = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/MRI-only XXX test data Cleaned')
# Set path to output folder with renamed data
outputPathRenamed = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/MRI-only XXX test data Cleaned and Renamed')

# Inference data extended
# Set path to raw data folder
rawDataInputPath = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/MRI-only XXX2 test data') 
# Set path to output folder with cleaned data
outputPathCleaned = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/MRI-only XXX2 test data Cleaned')
# Set path to output folder with renamed data
outputPathRenamed = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/MRI-only XXX2 test data Cleaned and Renamed')


# Make sure output folders exists
if not os.path.exists(outputPathCleaned):
    os.makedirs(outputPathCleaned)
if not os.path.exists(outputPathRenamed):
    os.makedirs(outputPathRenamed)

# Require folders to exist
imageFoldersMustExist = ['CT', 'MR']    
structsFoldersMustExist = ['RTSTRUCT']
otherFoldersMustExist = ['RTPLAN', 'RTDOSE']   
# Combine all folders that must exist
foldersMustExist = imageFoldersMustExist + structsFoldersMustExist + otherFoldersMustExist 

# Require MR series to contain this amount of slices
numSlicesMR = 88
# Require sCT series to contain this amount of slices
numSlicesCT = 86


### COPY DATA ###
print(' ')
print('Copying raw data')
print(' ')
# List patients in the raw data folder 
rawPatFolders = os.listdir(rawDataInputPath) 
# Get only directories
rawPatFolders = [x for x in rawPatFolders if os.path.isdir(os.path.join(rawDataInputPath, x))]
# Print amount of data
print('Amount of patients in raw data folder before selection: ', len(rawPatFolders))


### COPY DATA ###
# Loop over all patient folders and copy data to output folder
def patCopyRawData(patNr, patFolder):
    # Get patient folder path
    patFolderPathRaw = os.path.join(rawDataInputPath, patFolder)
    # Make sure folder path contains '19'
    assert '19' in patFolder, 'Patient folder name does not contain 19, it has no Pat ID'
    # Get patient ID from folder name
    patID = patFolder.split('_')[-1]
    # Copy data to output folder
    # Some patients have RT data and MR data split, this will fuse the data into the same folder
    # If patient directory exist do not copy
    if os.path.exists(os.path.join(outputPathCleaned, patID)) == False:
        shutil.copytree(patFolderPathRaw, os.path.join(outputPathCleaned, patID), dirs_exist_ok=True)

# Loop over patients
nrCPU = multiprocessing.cpu_count()-4
nrCPU = 60
# Init parallell job 
patInfo = Parallel(n_jobs=nrCPU, verbose=10)(delayed(patCopyRawData)(patNr, patFolder) for patNr, patFolder in enumerate(rawPatFolders))


### CLEAN DATA ###
# Now parallell loop over all patients in the output folder and check that all needed sub folders exist
# Also do other quality checks and remove data if it does not meet the defined criteria
# The function checks if data still exists or if it has been removed by the previous calls
print(' ')
print('Cleaning data as per criteria')
print(' ')
patFoldersToClean = os.listdir(outputPathCleaned)
# Get only directories
patFoldersToClean = [x for x in patFoldersToClean if os.path.isdir(os.path.join(outputPathCleaned, x))]

# Loop over patients
# for patFolder in patFolders:
def patCleanData(patNr, patFolder):
    # Print number
    print('Patient number: ', patNr)
    # Get patient folder path
    patFolderPath = os.path.join(outputPathCleaned, patFolder)

    if os.path.exists(patFolderPath) == True:
        ### FOLDER NAME AND DIGITS ###
        # Check that all characters in patient folder name are digits
        if patFolder.isdigit() == False:
            print('Current patient folder: ', patFolder)
            print('Patient folder name is not all digits: ', patFolder)
            print('Removing patient folder path: ', patFolderPath)
            shutil.rmtree(patFolderPath)
            
        # Check that patient folder name is 8 characters long
        if len(patFolder) != 12:
            print('Current patient folder: ', patFolder)
            print('Patient folder name is not 12 characters long: ', patFolder)
            print('Removing patient folder path: ', patFolderPath)
            shutil.rmtree(patFolderPath)
            
    # If patient folder still exists        
    if os.path.exists(patFolderPath) == True:
        ### SUB FOLDER FOR DATE ###
        patientDateFoldersCollect = []
        failFlag = False            
        # Check that all subfolders defined in foldersMustExist have only 1 subfolder in themselfs wich contains a folder name staring with '20'
        # Check that these sub folders have the same name (date) for all sub folders regarding image modality 
        for folder in foldersMustExist:
            # Get sub folder path
            subFolderPath = os.path.join(patFolderPath, folder)
            # If any of the required sub folders does not exist, remove patient folder (for data set completeness)
            if os.path.exists(patFolderPath) == True:
                if os.path.exists(subFolderPath) == False:
                    print('Current patient folder: ', patFolder)
                    print('Patient folder in output does not contain folder: ', folder)
                    # Write this stament but include the patuent patFolderPath
                    print('Patient folder path: ', patFolderPath)
                    print('Removing patient folder path: ', patFolderPath)
                    shutil.rmtree(patFolderPath)

            if os.path.exists(patFolderPath) == True:
                # Get all sub folders
                patientDateFolders = os.listdir(subFolderPath)
                # Get only directories
                patientDateFoldersFound = [x for x in patientDateFolders if os.path.isdir(os.path.join(subFolderPath, x))]
                # Assert that there are only 1 sub folder, else remove patient folder
                if len(patientDateFoldersFound) != 1: 
                    print('Current patient folder: ', patFolder)
                    print('Patient has had multiple scanning dates: ', patientDateFolders)
                    print('Removing patient folder path: ', patFolderPath)
                    shutil.rmtree(patFolderPath)

            # If patient folder still exists            
            if os.path.exists(patFolderPath) == True:
                # Add it to a list of folders patientDateFolders
                patientDateFoldersCollect.append(patientDateFoldersFound[0])
                # Make sure all list items starts with '20'
                for dateFolder in patientDateFoldersCollect:
                    assert dateFolder.startswith('20'), 'Sub folder does not start with 20 for date: ' + folder

                # Make sure that all the sub date folders are the same, if not remove patient folder
                if len(set(patientDateFolders)) != 1: 
                    print('Current patient folder: ', patFolder)
                    print('Sub folders do not contain the same date: ', patientDateFolders)
                    print('Removing patient folder path: ', patFolderPath)
                    shutil.rmtree(patFolderPath)

        # If patient folder still exists
        if os.path.exists(patFolderPath) == True:
            ### SUB FOLDER FOR IMAGE MODALITIES ###
            for folder in foldersMustExist:
                # Get sub folder path
                subFolderPath = os.path.join(patFolderPath, folder) 
                if os.path.exists(patFolderPath) == True:         
                    # Check number of files in sub folders
                    # Get number of files in sub folder
                    if folder == 'RTPLAN' or folder == 'RTDOSE':
                        numFiles, filePaths = count_files_in_dir(subFolderPath) 
                        if numFiles != 1:
                            print('Current patient folder: ', patFolder)
                            print('Sub folder does not contain exactly 1 file: ', folder)
                            print('Removing patient folder path: ', patFolderPath)
                            shutil.rmtree(patFolderPath)
 

                # If patient folder still exists
                if os.path.exists(patFolderPath) == True:
                    # Check number of folders in MR folder. 
                    # Remove patients that have been rescanned multiple times
                    # Get number of files in sub folder
                    if folder == 'MR': 
                        # Get the date folder path within MR folder
                        dateFolderPath = os.path.join(patFolderPath, 'MR', patientDateFoldersCollect[0])
                        numDirs = count_dirs_in_dir(dateFolderPath) 
                        if numDirs != 1:
                            print('Current patient folder: ', dateFolderPath)
                            print('Sub folder under date folder does not contain exactly 1 dir: ', folder)
                            print('Removing patient folder path: ', patFolderPath)
                            shutil.rmtree(patFolderPath)

                        # Check that we have 88 slices in the MR folder
                        if numDirs == 1:
                            # Get the MR folder path
                            MRFileFolderPath = os.path.join(dateFolderPath, os.listdir(dateFolderPath)[0])
                            # Get the number of files in the MR folder
                            numFiles, filePaths = count_files_in_dir(MRFileFolderPath) 
                            if numFiles != numSlicesMR:
                                print('Current patient folder: ', patFolder)
                                print('MR Sub folder under date folder does not contain exactly required number of files: ', MRFileFolderPath)
                                print('Removing patient folder path: ', patFolderPath)
                                shutil.rmtree(patFolderPath)

                            # If patient still exists
                            if os.path.exists(patFolderPath) == True:
                                # Check MR slice thickness and slice consistency
                                QAcheckMR = checkImageVolume(MRFileFolderPath)
                                
                                if QAcheckMR == False:
                                    print('Current patient folder: ', patFolder)
                                    print('MR data has inconsistencies: ', MRFileFolderPath)
                                    print('Removing patient folder path: ', patFolderPath)
                                    shutil.rmtree(patFolderPath)

                
                # If patient folder still exists
                if os.path.exists(patFolderPath) == True:
                    # Check DICOM data in CT folder
                    # Get number of files in sub folder
                    if folder == 'CT': 
                        # Get the date folder path within CT folder
                        dateFolderPath = os.path.join(patFolderPath, 'CT', patientDateFoldersCollect[0])
                        numDirs = count_dirs_in_dir(dateFolderPath) 
                        if numDirs != 1:
                            print('Current patient folder: ', dateFolderPath)
                            print('Sub folder under date folder does not contain exactly 1 dir: ', folder)
                            print('Removing patient folder path: ', patFolderPath)
                            shutil.rmtree(patFolderPath)
                            
                        if numDirs == 1:
                            # Get the CT folder path
                            CTFileFolderPath = os.path.join(dateFolderPath, os.listdir(dateFolderPath)[0])
                            # Check sCT slice thickness and slice consistency
                            QAchecksCT = checkImageVolume(CTFileFolderPath)
                            if QAchecksCT == False:
                                print('Current patient folder: ', patFolder)
                                print('sCT data has inconsistencies: ', CTFileFolderPath)
                                print('Removing patient folder path: ', patFolderPath)
                                shutil.rmtree(patFolderPath)
                                

                            
                # If patient folder still exists
                if os.path.exists(patFolderPath) == True:                
                    if folder == 'RTSTRUCT':
                        numFiles, filePaths = count_files_in_dir(subFolderPath)                    
                        # There should be exactly 2 RT struct files, one for contorus and one for fiducial markers
                        if numFiles != 2:
                            print('Current patient folder: ', patFolder)
                            print('Sub folder does now contain exactly 2 RT struct files: ', folder)
                            print('Removing patient folder path: ', patFolderPath)
                            shutil.rmtree(patFolderPath)
                            
                        # Even if there are 2 files, they should be contained in 2 different sub folders with specific names
                        if numFiles == 2:
                            if 'Contouring' not in filePaths[0] and 'Contouring' not in filePaths[1]:
                                print('Current patient folder: ', patFolder)
                                print('Sub folder does not contain any contouring folder: ', folder)
                                print('Removing patient folder path: ', patFolderPath)
                                shutil.rmtree(patFolderPath)

                            if os.path.exists(patFolderPath) == True:     
                                if 'ARIA_RadOnc_Structure_Sets' not in filePaths[0] and 'ARIA_RadOnc_Structure_Sets' not in filePaths[1]:
                                        print('Current patient folder: ', patFolder)
                                        print('Sub folder does not contain any ARIA_RadOnc_Structure_Sets folder: ', folder)
                                        print('Removing patient folder path: ', patFolderPath)
                                        shutil.rmtree(patFolderPath)
                                    
                            if os.path.exists(patFolderPath) == True:  
                                # Check that the file size of the RT struct files are reasonable 
                                # with regards to the image series it is referencing
                                for filePath in filePaths:
                                    if os.path.exists(patFolderPath) == True: 
                                        if 'ARIA_RadOnc_Structure_Sets' in filePath: # Full contour file
                                            if os.path.getsize(filePath) < 1024*1000: # 1 MB
                                                print('Current patient folder: ', patFolder)
                                                print('Full contour file size seems to small: ', filePath)
                                                print('Removing patient folder path: ', patFolderPath)
                                                shutil.rmtree(patFolderPath)
                                                
                                        if 'Contouring' in filePath: # Fiducial marker file 
                                            # Check file size
                                            if os.path.getsize(filePath) > 1024*14: # 14 kb
                                                print('Current patient folder: ', patFolder)
                                                print('Fiducial contour file size seems to big: ', filePath)
                                                print('Removing patient folder path: ', patFolderPath)
                                                shutil.rmtree(patFolderPath)
                                                              
# Loop over patients
nrCPU = multiprocessing.cpu_count()-4
#nrCPU = 60
# Init parallell job 
patInfo = Parallel(n_jobs=nrCPU, verbose=10)(delayed(patCleanData)(patNr, patFolder) for patNr, patFolder in enumerate(patFoldersToClean))


### COPY DATA TO FINAL FOLDER AND DO QA ### 
# Copy data to another location but change sub folder names to make it all homogenous
# Also change the patient id to a random number. DICOM data is still NOT anonymized but the 
#changed patient folder ID will be inherited in the Nifti and nnUnet training data.
print(' ')
print('Copying data and performing QA checks')
print(' ')
# List patients in the raw data folder 
patFolders = os.listdir(outputPathCleaned) 
# Get only directories
patFolders = [x for x in patFolders if os.path.isdir(os.path.join(outputPathCleaned, x))]
# Print amount of data
print('Amount of patients in cleaned data folder before renaming: ', len(patFolders))

def patCopyQA(patNr, patFolder):
    # Create new random seed
    # Important for parallell threading
    R=np.random.RandomState()
    # Create random large integer
    # Use it to name the folders
    randPatValue = R.randint(100000000000, 999999999999)
    
    # Get patient folder path
    patFolderPath = os.path.join(outputPathCleaned, patFolder)
    # Get path of a sub folder which has the specifc keyword in it
    SyntheticCTFolderPath = find_folder(patFolderPath, 'Synthetic_CT_with_fiducial_markers')
    StorT2FolderPath = find_folder(patFolderPath, 'Stor_T2_till_sCT')
    DoseFolderPath = find_folder(patFolderPath, 'Eclipse_Doses')
    PlanFolderPath = find_folder(patFolderPath, 'ARIA_RadOnc_Plans')
    StructureContourFolderPath = find_folder(patFolderPath, 'ARIA_RadOnc_Structure_Sets')
    FiducialStructFolderPath = find_folder(patFolderPath, 'Contouring')   

    # Check what MR protocol version that has been used for the MR data.
    MRProtocolAcqVersion = checkMRAcqProtocol(StorT2FolderPath)
    patFolderNewAnonID = MRProtocolAcqVersion + '_' + str(randPatValue)
    
    # Copy each sub folder to a new location with a new name based of patFolder
    # Synthetic CT
    # Create destination folder
    renamed_SyntheticCTFolderPath = os.path.join(outputPathRenamed, patFolderNewAnonID, 'sCT')
    shutil.copytree(SyntheticCTFolderPath, renamed_SyntheticCTFolderPath, dirs_exist_ok=True)
    # Stor T2
    renamed_StorT2FolderPath = os.path.join(outputPathRenamed, patFolderNewAnonID, 'MR_StorT2')
    shutil.copytree(StorT2FolderPath, renamed_StorT2FolderPath, dirs_exist_ok=True)
    # Dose
    renamed_DoseFolderPath = os.path.join(outputPathRenamed, patFolderNewAnonID, 'DoseDist')
    shutil.copytree(DoseFolderPath, renamed_DoseFolderPath, dirs_exist_ok=True)
    # Plan
    renamed_PlanFolderPath = os.path.join(outputPathRenamed, patFolderNewAnonID, 'DosePlan')
    shutil.copytree(PlanFolderPath, renamed_PlanFolderPath, dirs_exist_ok=True)
    # sCT contours
    renamed_StructureContourFolderPath = os.path.join(outputPathRenamed, patFolderNewAnonID, 'RTStruct', 'sCTContours')
    shutil.copytree(StructureContourFolderPath, renamed_StructureContourFolderPath, dirs_exist_ok=True)
    # FiducialStruct
    renamed_FiducialStructFolderPath = os.path.join(outputPathRenamed, patFolderNewAnonID, 'RTStruct', 'fiducialStorT2')
    shutil.copytree(FiducialStructFolderPath, renamed_FiducialStructFolderPath, dirs_exist_ok=True)

    ### QA ###
    # Make sure there are 86 files in sCT folder
    numFiles, filePaths = count_files_in_dir(renamed_SyntheticCTFolderPath)
    if numFiles < numSlicesCT:
        print('Number of files in sCT folder: ', numFiles)
        print('Patient folder: ', patFolder)

    # Make sure there are 88 files in MR_StorT2 folder
    numFiles, filePaths = count_files_in_dir(renamed_StorT2FolderPath)
    if numFiles < numSlicesMR: 
        print('Number of files in MR_StorT2 folder: ', numFiles)
        print('Patient folder: ', patFolder)

    # Make sure there are 1 files in DoseDist folder
    numFiles, filePaths = count_files_in_dir(renamed_DoseFolderPath)
    assert numFiles == 1, 'There are not 1 files in the DoseDist folder'
    # Make sure there are 1 files in DosePlan folder
    numFiles, filePaths = count_files_in_dir(renamed_PlanFolderPath)
    assert numFiles == 1, 'There are not 1 files in the DosePlan folder'
    # Make sure there are 1 file in sCTContours folder
    numFiles, filePaths = count_files_in_dir(renamed_StructureContourFolderPath)
    assert numFiles == 1, 'There are not 1 files in the sCTContours folder'
    # Make sure there are 1 file in fiducialStorT2 folder
    numFiles, filePaths = count_files_in_dir(renamed_FiducialStructFolderPath)
    assert numFiles == 1, 'There are not 1 files in the fiducialStorT2 folder'

    # Check slice thickness and slice consistency
    # Check sCT
    checkImageVolume(renamed_SyntheticCTFolderPath)
    # Check MR
    checkImageVolume(renamed_StorT2FolderPath)

    # Check DICOM UIDs
    # Get MR series instance UID and frame of reference UID
    MRseriesInstanceUID = getSeriesInstanceUID(renamed_StorT2FolderPath)
    MRframeOfReferenceUID = getFrameOfReferenceUID(renamed_StorT2FolderPath)
    # Get sCT series instance UID and frame of reference UID
    sCTseriesInstanceUID = getSeriesInstanceUID(renamed_SyntheticCTFolderPath)
    sCTframeOfReferenceUID = getFrameOfReferenceUID(renamed_SyntheticCTFolderPath)
    # Get fiducial referenced series instance UID
    referencedfiducialSeriesInstanceUID = referencedSeriesInstanceUID(renamed_FiducialStructFolderPath)
    # Get sCT contour referenced series instance UID
    referencedsCTcontourSeriesInstanceUID = referencedSeriesInstanceUID(renamed_StructureContourFolderPath)
    # Assert that the MR and sCT frame of reference UID are the same
    assert MRframeOfReferenceUID == sCTframeOfReferenceUID, 'MR and sCT frame of reference UID are not the same'
    # Assert that the MR and fiducial series instance UID are the same
    assert MRseriesInstanceUID == referencedfiducialSeriesInstanceUID, 'MR and fiducial series instance UID are not the same'
    # Assert that the sCT and sCT contour series instance UID are the same
    assert sCTseriesInstanceUID == referencedsCTcontourSeriesInstanceUID, 'sCT and sCT contour series instance UID are not the same'

    # Return MRProtocolAcqVersion
    return (patFolder, patFolderNewAnonID, MRProtocolAcqVersion) 


# Loop over patients
#nrCPU = multiprocessing.cpu_count()-4
nrCPU = 60
# Init parallell job 
patInfo = Parallel(n_jobs=nrCPU, verbose=10)(delayed(patCopyQA)(patNr, patFolder) for patNr, patFolder in enumerate(patFolders))


# Write MRProtocolAcqVersion statistics to a text file
with open('/mnt/mdstore2/Christian/MRIOnlyData/MRProtocolAcqVersionStats.txt', 'w') as f:
    for patFolder, patFolderNewAnonID, MRProtocolAcqVersion in patInfo:
        f.write("%s,%s\n" % (patFolderNewAnonID,MRProtocolAcqVersion)) 

# Write code list to convert patFolderNewAnonID to patFolder
with open('/mnt/mdstore2/Christian/MRIOnlyData/NewAnonID_to_patFolder.txt', 'w') as f:
    for patFolder, patFolderNewAnonID, MRProtocolAcqVersion in patInfo:
        f.write("%s,%s\n" % (patFolderNewAnonID,patFolder)) 


# Read the MRProtocolAcqVersionStats file
with open('/mnt/mdstore2/Christian/MRIOnlyData/MRProtocolAcqVersionStats.txt', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

# Flatten the list and count occurrences
data = [item for sublist in data for item in sublist]
counter = Counter(data)
print('Total number of patients: ', len(patFolders))
print('Total number of patients with the old MR acq protocol: ', counter['oldAcq'])
print('Total number of patients with the new MR acq protocol: ', counter['newAcq'])

# Append statistics to a text file
with open('/mnt/mdstore2/Christian/MRIOnlyData/MRProtocolAcqVersionStats.txt', 'a') as f:
    f.write("%s, %s\n" % ('Total number of patients', len(patFolders)))
    f.write("%s, %s\n" % ('Total number of patients with the old MR acq protocol', counter['oldAcq']))
    f.write("%s, %s\n" % ('Total number of patients with the new MR acq protocol', counter['newAcq']))

    