# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Sets a new SeriesInstanceUID every file in a DICOM series.
# Reason: I had to correct this as my old SeriesInstanceUID was not unique for each series. 
# It was based on time points and I did inference in parallell. Therefore I got
# SeriesInstanceUIDs that were the same for different series. This is not allowed and I am 
# not able to import that data to Eclipse. This function outputs the data in a new folder.
# SeriesInstanceUID is now a unique identifier for each series.
# *********************************************************************************

import os
import numpy as np
import pydicom
import shutil

def correctSeriesInstanceUID(inputDirPath, outputDirName):
    """
    Corrects the SeriesInstanceUID of all DICOM files in a folder. 
    The new SeriesInstanceUID is based on random. 
    The new files are saved in a new sub folder with the name outputDirName. 
    """
    # Set full path for new folder
    outputDirPath = os.path.join(inputDirPath, outputDirName)

    # If folder already exist, remove it
    if os.path.exists(outputDirPath):
        # Print
        print('Removing existing folder:', outputDirPath)
        shutil.rmtree(outputDirPath)

    # Make sure output folder exist
    if not os.path.exists(outputDirPath):
        os.makedirs(outputDirPath)

    # Set a new SeriesInstanceUID from generate_uid().
    # Should be set to same for all DICOM files as this is a series
    newSeriesInstanceUID = pydicom.uid.generate_uid()
    # Loop over all files in the input directory
    for filename in os.listdir(inputDirPath):
        # Disregard folders
        if os.path.isdir(os.path.join(inputDirPath, filename)):
            continue
        # Assert it is a DICOM file
        assert filename.endswith('.dcm')
        # Print the filename
        print('Processing file:', filename)
        # Read the DICOM file
        ds = pydicom.dcmread(os.path.join(inputDirPath, filename))
        # Set a new SeriesInstanceUID 
        ds.SeriesInstanceUID = newSeriesInstanceUID
        # Save the file in the new folder
        ds.save_as(os.path.join(outputDirPath, filename))
        # If successful, remove the source file if removeFile is set to True
        if os.path.exists(os.path.join(outputDirPath, filename)): # Check if file was saved correctly
            if removeFile: # Remove source file
                os.remove(os.path.join(inputDirPath, filename))



"""
# Old entry point with a single folder
# Entry point of the script
if __name__ == "__main__":
    ### Config ###
    # Set the path to the folder containing the DICOM series

    inputDirPath = os.path.join(r'\\XXX\va_transfer\Temp\Temp Jonas CJG\ChristianUncObsEclipseForStep2\test_unc_pat34_obsD\Step2\test_unc_pat34_obsD_111_uncertaintyMap_DICOM')
    outputDirName = 'newSeriesInstanceUID'
    # Call the function to correct the SeriesInstanceUID
    correctSeriesInstanceUID(inputDirPath, outputDirName)
    print('Done for the folder:', inputDirPath)
    print('The new files are saved in the folder:', os.path.join(inputDirPath, outputDirName))
"""

# Entry point of the script. Enabling batch processing of multiple folders
if __name__ == "__main__":
    ### Config ###
    baseDir = os.path.join(r'\\XXX\va_transfer\Temp\Temp Jonas CJG\ChristianUncObsEclipseForStep2')
    outputDirName = 'newSeriesInstanceUID'
    obs = 'obsE'  # Config variable for observer
    obs_values = [110, 111]  # Values to be appended
    # Remove source file after changing SeriesInstanceUID
    removeFile = True

    # Outer loop for patients
    for patNum in range(1, 36):
        patStr = f'pat{patNum}'

        # Inner loop for test directory names
        for obsNum in obs_values:
            subjectDirName = f'test_unc_{patStr}_{obs}'
            inputDirName = f'test_unc_{patStr}_{obs}_{obsNum}_uncertaintyMap_DICOM'
            inputDirPath = os.path.join(baseDir, subjectDirName, 'Step2', inputDirName)

            # Call the function to correct the SeriesInstanceUID
            print('Processing folder:', inputDirPath)
            correctSeriesInstanceUID(inputDirPath, outputDirName)
            print('Done for the folder:', inputDirPath)
            print('The new files are saved in the folder:', os.path.join(inputDirPath, outputDirName))