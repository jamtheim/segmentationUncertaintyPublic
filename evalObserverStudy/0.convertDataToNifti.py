# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Script for converting DICOM RT struct observer data to Nifti format. 
# Nifti format allows for further evaluation of the data. 
# *********************************************************************************

import os
import shutil
import numpy as np
import SimpleITK as sitk
from commonConfig import commonConfigClass
from evalDataMethods import evalDataMethodsClass
# Load configuration
conf = commonConfigClass() # Configuration for the project
# Load methods
evalData = evalDataMethodsClass() # Functions for converting DICOM to Nifti data

# This is the logic structure for the script
# Loop through each observer data folder
    # Loop through each patient 
        # Loop through step1Edited and step2Edited
            # Loop through CTV1/2 and Rectum1/2
                # Convert DICOM to Nifti and save in the same folder as the DICOM data
                    # Asses that correct amount of files are present in the output folder

# Loop through each observer data folder
for observer in conf.eval.observers:
    # Loop through each patient 
    for patient in conf.eval.patients:
        print('Reference data for observer ' + observer + ' and patient ' + patient)
        # Define the reference DICOM image folder for each patient. It is needed to convert RT structure to Nifti file. It provides the spacing and orientation of the image. 
        dicomReferenceFolderPath = os.path.join(conf.base.infDataBaseFolderPath, conf.base.infObsFolderName + '_' + observer, conf.base.infEclipseDataFolderName, patient + '_' + observer, conf.base.infStep1FolderName, conf.base.infImageSeriesName)
        
        # Copy the reference AI data for the observer data to the observer data folder, converted as Nifti data.
        # Original reference AI data should be DICOM data that originated from Nifti from nnUnet, converted to RT struct, 
        # imorted into Eclipse and then exported as DICOM RT struct again. Not doing so introduces interpolation errors.
        # This was done at the second data import step, i.e the structure was exported from Eclipse after import.  
        # There seem to be small interpolation errors when using the original nnUNet Nifti file and along the paths of converting. Therefore avoid using this raw data. 
        # By using data from Eclipse, both in terms of reference and edited data we make sure that we are not being affected by interpolation errors.

        # Reference RT struct data. This is the original AI model output, imported into Eclipse and exported as DICOM RT struct.
        # CTV
        CTV_RTStructReferenceFolderPath = os.path.join(conf.base.obsDataEditedFolderPath, observer, patient + '_' + observer, conf.base.referenceFolderName, 'CTV')
        CTV_RTStructReferenceFile = evalData.getRTStructFile(CTV_RTStructReferenceFolderPath)
        CTV_RTStructReferenceFilePath = os.path.join(CTV_RTStructReferenceFolderPath, CTV_RTStructReferenceFile)
        # QA the reference data so contained structure in RT struct really is correct
        if conf.eval.QA == True:
            evalData.qaRTStruct(observer, patient, CTV_RTStructReferenceFilePath, 0, 'ref', 'CTV')

        # Rectum
        Rectum_RTStructReferenceFolderPath = os.path.join(conf.base.obsDataEditedFolderPath, observer, patient + '_' + observer, conf.base.referenceFolderName, 'Rectum')
        Rectum_RTStructReferenceFile = evalData.getRTStructFile(Rectum_RTStructReferenceFolderPath)
        Rectum_RTStructReferenceFilePath = os.path.join(Rectum_RTStructReferenceFolderPath, Rectum_RTStructReferenceFile)
        # QA the reference data so contained structure in RT struct really is correct
        if conf.eval.QA == True:
            evalData.qaRTStruct(observer, patient, Rectum_RTStructReferenceFilePath, 0, 'ref', 'Rectum')
        # Create a reference Nifti folder in each patient folder for each structure. This will contain the reference converted to Nifti data.
        nifti_CTV_RTStructReferenceFolderPath = os.path.join(CTV_RTStructReferenceFolderPath, conf.base.niftiFolderName)
        nifti_Rectum_RTStructReferenceFolderPath = os.path.join(Rectum_RTStructReferenceFolderPath, conf.base.niftiFolderName)

        # If Nifit folder exists, remove the folders and create a new one.
        # This is to make sure that the folder is empty and that the new data is not appended to the old data in the old folder
        # CTV
        if os.path.exists(nifti_CTV_RTStructReferenceFolderPath):
            print('Removing existing reference Nifti folder for CTV: ' + nifti_CTV_RTStructReferenceFolderPath)
            shutil.rmtree(nifti_CTV_RTStructReferenceFolderPath)
        # Rectum 
        if os.path.exists(nifti_Rectum_RTStructReferenceFolderPath):
            print('Removing existing reference Nifti folder for Rectum: ' + nifti_Rectum_RTStructReferenceFolderPath)
            shutil.rmtree(nifti_Rectum_RTStructReferenceFolderPath)

        # Create the folders again
        # CTV
        os.makedirs(nifti_CTV_RTStructReferenceFolderPath, exist_ok=True)
        # Rectum
        os.makedirs(nifti_Rectum_RTStructReferenceFolderPath, exist_ok=True)

        # Print information 
        #print('Reference Nifti CTV folder: ' + nifti_CTV_RTStructReferenceFolderPath)
        #print('Reference Nifti Rectum folder: ' + nifti_Rectum_RTStructReferenceFolderPath)

        print('Converting reference DICOM to Nifti for CTV: ' + CTV_RTStructReferenceFilePath)
        # Convert DICOM to Nifti (last function argument sets naming of final Nifti file, step set to 0 for reference data)
        CTVNiftiReferenceFilePath = evalData.DicomRT2Nifti(CTV_RTStructReferenceFilePath, dicomReferenceFolderPath, nifti_CTV_RTStructReferenceFolderPath, observer, patient, 'CTV', 0, 'refData')
        # Assert that only one file is present in the folder
        evalData.assertOnlyOneNiftiFile(nifti_CTV_RTStructReferenceFolderPath)

        print('Converting reference DICOM to Nifti for Rectum: ' + Rectum_RTStructReferenceFilePath)
        # Convert DICOM to Nifti (last argument sets naming of final Nifti file, step set to 0 for reference data)
        RectumNiftiReferenceFilePath = evalData.DicomRT2Nifti(Rectum_RTStructReferenceFilePath, dicomReferenceFolderPath, nifti_Rectum_RTStructReferenceFolderPath, observer, patient, 'Rectum', 0, 'refData')
        # Assert that only one file is present in the folder
        evalData.assertOnlyOneNiftiFile(nifti_Rectum_RTStructReferenceFolderPath)

        # Continue with the conversion of observer data (edited data)
        # Loop through step1Edited and step2Edited
        for step in conf.eval.steps:
            print('')
            print('Observer data for observer ' + observer + ' and patient ' + patient + ' for step ' + str(step))
            # Loop through CTV and Rectum
            for structure in conf.eval.structures:
                # Print the structure
                print('Structure: ' + structure)
                # Convert DICOM to Nifti and save in the same folder as the DICOM data
                dicomFolder = os.path.join(conf.base.infDataBaseFolderPath, conf.base.obsDataEditedBaseFolder, observer, patient + '_' + observer, 'step' + str(step) + 'Edited', structure + str(step))
                # Check that only one dicom file is present in the folder
                dicomStructFile = evalData.getRTStructFile(dicomFolder)
                # Get full file path
                dicomStructFilePath = os.path.join(dicomFolder, dicomStructFile)
                print(dicomStructFilePath)
                # QA the structure RT struct data so contained structure in RT struct really is correct
                if conf.eval.QA == True: 
                    evalData.qaRTStruct(observer, patient, dicomStructFilePath, step, 'obs', structure)
                # Create a Nifti folder in each structure folder
                niftiFolderPath = os.path.join(dicomFolder, conf.base.niftiFolderName)
                # If existing, remove the folder and create a new one
                # This is to make sure that the folder is empty and that the new data is not appended to the old data
                if os.path.exists(niftiFolderPath):
                    print('Removing existing Nifti folder for: ' + niftiFolderPath)
                    shutil.rmtree(niftiFolderPath)
                # Create the folder again
                os.makedirs(niftiFolderPath, exist_ok=True)
                print('Created nifti folder: ' + niftiFolderPath)
                # Convert DICOM to Nifti
                # print('Reference DICOM folder: ' + dicomReferenceFolderPath)
                print('Converting DICOM to Nifti observer data for: ' + dicomStructFilePath)
                # Convert DICOM to Nifti (last argument desides naming of final Nifti file)
                convertedNiftiFilePath = evalData.DicomRT2Nifti(dicomStructFilePath, dicomReferenceFolderPath, niftiFolderPath, observer, patient, structure, step, 'obsData')
                # Assert that only one file is present in the folder
                evalData.assertOnlyOneNiftiFile(niftiFolderPath)
            # Print empty line
            print('')