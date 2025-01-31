# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Configuration file for the data pipeline. 
# Controls both the pre-processing, training and inference.
# *********************************************************************************

import os
import numpy as np


class commonConfigClass():
    """
    Class describing the common configuration used in the project.
    """

    def __init__ (self, inferenceSubFolderInput=None, inferenceTaskNumberInput=None):
        """
        Initialize the common configuration class.

        Args:
            inferenceSubFolder (str, optional): The subfolder for inference. Defaults to None.
        """
        if inferenceSubFolderInput is not None:
            print("inferenceSubFolderInput:", inferenceSubFolderInput)  # Print the inference subfolder
            print("inferenceTaskNumberInput:", inferenceTaskNumberInput)  # Print the inference task number
        # Assign the initialized configurations to instance variables
        self.base, self.preProcess, self.inference = self.initializeConfiguration(inferenceSubFolderInput, inferenceTaskNumberInput)

        # Print all variables in self.inference
        #for key, value in self.inference.__dict__.items():
        #    print(key, ' : ', value)



    class base:
        """
        Empty class to define base related configuration.
        """
        pass

    class preProcess:
        """
        Empty class to define preProcessing related configuration.
        """
        pass

    class inference:
        """
        Empty class to define inference related configuration.
        """
        pass



    def initializeConfiguration(self, inferenceSubFolderInput, inferenceTaskNumberInput):
        """
        Initialize the configurations based on the provided inference folder.

        Args:
            inferenceSubFolder (str): The subfolder for inference.

        Returns:
            tuple: A tuple containing the base, preProcess, and inference configurations.
        """
        # Make sure inferenceTaskNumberInput is an integer
        if inferenceTaskNumberInput is not None:
            inferenceTaskNumberInput = int(inferenceTaskNumberInput)

        # Set base configuration
        base = self.base()
        preProcess = self.preProcess()
        inference = self.inference()
    
        # Set base configuration
        # Working folder 
        base.workFolder = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/')

        # Set body struct name
        base.bodyStructureName1 = 'BODY'
        base.bodyStructureName2 = 'body'
        # Set preProcess configuration
        preProcess.nnUNetTaskNumber = 100
        preProcess.nnUNetTaskName = 'Dataset' + str(preProcess.nnUNetTaskNumber)
        # Folder where all patient folders are located
        preProcess.inputDicomPatientDir = os.path.join(base.workFolder, 'MRI-only from DATE Cleaned and Renamed') #ID/CT and ID/RTstruct
        # Make sure folder exist 
        #os.makedirs(preProcess.inputDicomPatientDir, exist_ok=True)
        # Folder where all patient folders converted to Nifti will be stored
        preProcess.outputNiftiPatientDir = os.path.join(base.workFolder, 'MRI-only from DATE Cleaned and Renamed Nifti')
        # Make sure folder exist
        os.makedirs(preProcess.outputNiftiPatientDir, exist_ok=True)
        # Folder where the final training data will be stored
        preProcess.outputTrainingDataDir = os.path.join(base.workFolder, 'MRI-only from DATE Cleaned and Renamed Nifti final', preProcess.nnUNetTaskName)
        # Make sure folder exist
        os.makedirs(preProcess.outputTrainingDataDir, exist_ok=True)
        preProcess.TrainingDataImageDir = 'imagesTr'
        preProcess.TrainingDataLabelDir = 'labelsTr'
        # Folder where CT data will be stored
        preProcess.CTfolder = 'MR_StorT2'
        # Folder where RT struct data will be stored
        preProcess.RTstructFolder = 'RTStruct'

        # If there is a need to train model on multiple structures, where structures are not allowed to overlap
        # Different configurations for different tasks can be set here
        # Define dictionary with structure name and encoded value
        # Background is always equal to value 0
        # Dictionary must start with 1 and increase +1 for each structure added
        # Idea of using multiple separate models to avoid structure overlap 
        if preProcess.nnUNetTaskNumber == 100: 
            preProcess.structureOfInterestDict = {'CTVT_427': 1}
        if preProcess.nnUNetTaskNumber == 101: 
            preProcess.structureOfInterestDict = {'Rectum': 1}
        if preProcess.nnUNetTaskNumber == 110: 
            preProcess.structureOfInterestDict = {'CTVT_427_10': 1}
        if preProcess.nnUNetTaskNumber == 111: 
            preProcess.structureOfInterestDict = {'Rectum_10': 1}

        
        # Define used prefix for outputed structures
        preProcess.fileMaskPrefix = 'mask_'
        # Define use file suffix
        preProcess.fileSuffix = '.nii.gz'
        # Define image volume name
        preProcess.imageVolumeName = 'image.nii.gz'

        # Load inference sub folder dynamically from input
        inference.inferenceSubFolder = inferenceSubFolderInput
        
        # If there is a defined folder for inference
        if inference.inferenceSubFolder is not None:
            # Task number for nnUNet
            inference.nnUNetTaskNumber = inferenceTaskNumberInput
            # Number of models used in the specific task number above
            if inference.nnUNetTaskNumber == 100: # Prostate CTV 5 models
                inference.numberOfModels = 5
                inference.structureOfInterestDict = {'CTVT_427': 1}
            if inference.nnUNetTaskNumber == 101: # Prostate Rectum 5 models
                inference.numberOfModels = 5
                inference.structureOfInterestDict = {'Rectum': 1}
            if inference.nnUNetTaskNumber == 110: # Prostate CTV 10 models
                inference.numberOfModels = 10
                inference.structureOfInterestDict = {'CTVT_427_10': 1}
            if inference.nnUNetTaskNumber == 111: # Prostate Rectum 10 models
                inference.numberOfModels = 10
                inference.structureOfInterestDict = {'Rectum_10': 1}

            # Set inference configuration  
            inference.inputDicomPatientDir = os.path.join(base.workFolder, 'inference', inference.inferenceSubFolder, 'inference_DICOM_in')
            # Set inference DICOM folder (static to MRI-only_test_data as this is the only data used for inference in this project)
            #inference.inputDicomPatientDir = os.path.join(base.workFolder, 'inference', 'MRI-only_test_data', 'inference_DICOM_in')
            
            # Set folder for sample data (obsolete, not used in this project)
            # inference.inputDicomPatientDirSampleData = os.path.join(base.workFolder, 'inference', inference.inferenceSubFolder, 'sampleDataDICOM')
            # Make sure folder exist
            os.makedirs(inference.inputDicomPatientDir, exist_ok=True)
            # Set inference Nifti folder
            inference.inputNiftiPatientDir = os.path.join(base.workFolder, 'inference', inference.inferenceSubFolder, 'inference_NIFTI_in')
            # Make sure folder exist
            os.makedirs(inference.inputNiftiPatientDir, exist_ok=True)
            # Define export folder for data that is going to be forded to Varian Eclipse
            inference.exportEclipseDir = os.path.join(base.workFolder, 'inference', inference.inferenceSubFolder, 'exportEclipse')
            # Make sure folders exist
            os.makedirs(inference.exportEclipseDir, exist_ok=True)
            
            # Set inference output folder for nnUNet
            inference.outputNiftiSegPatientDir = os.path.join(base.workFolder, 'inference', inference.inferenceSubFolder, 'inference_NiftiSeg_out' + '_TaskNumber_' + str(inference.nnUNetTaskNumber))
            # Make sure folder exist
            os.makedirs(inference.outputNiftiSegPatientDir, exist_ok=True)
            # Set output base folder for RT structure data
            inference.outputRTstructPatientDirBase = os.path.join(base.workFolder, 'inference', inference.inferenceSubFolder, 'inference_RTstruct_out')
            # Set output folder for RT structure data
            inference.outputRTstructPatientDir = os.path.join(base.workFolder, 'inference', inference.inferenceSubFolder, 'inference_RTstruct_out'+ '_TaskNumber_' + str(inference.nnUNetTaskNumber))
            # Make sure folder exist
            os.makedirs(inference.outputRTstructPatientDir, exist_ok=True)
            # Below are dictinaries for the inference. The keys are the names of the structures
            # defined in pre-processing and must be mathed. 
            # Define dictionary for setting new inference name
            # Current structure Id / New structure Id
            inference.inferenceNameDict = {'Bladder': 'Bladder', 'femur_head_r': 'Femoral_Head_R', 'femur_head_l': 'Femoral_Head_L', 'Rectum': 'Rectum_AI', 'Rectum_10': 'Rectum_10_AI', 'CTVT_427': 'CTVT_427_AI', 'CTVT_427_10': 'CTVT_427_10_AI', 'Heart_Ref': 'Heart_AI', 'Eso_Ref': 'Eso_AI', 'ProxBronch_Ref': 'ProxBronch_AI'}
            # Define dictionary for defining RGB codes
            inference.inferenceRGBcodeDict = {'bladder': [255, 255, 0], 'Bladder': [255, 255, 0], 'Rectum': [102, 51, 0], 'Rectum_10': [102, 51, 0], 'CTVT_427': [102, 51, 0], 'CTVT_427_10': [102, 51, 0]}
            
            # Set number of GPUs to use for inference
            # Can only be 1 for softmax output for every fold. Not adapted to more GPUs yet but can be done.
            # Rather use multiple models for inference in parallel
            inference.nrGPU = 1 

            # Fusion of RT structures from DICOM files. Define task numbers (models)
            inference.fuseStructsFromTasks = [600, 500, 502]
            # Set output temp folder for RT structure data
            inference.outputRTstructPatientStage1Temp = os.path.join(base.workFolder, 'inference', inference.inferenceSubFolder, 'nnUNet_data', 'inference_RTstruct_Stage1Temp')
            # Make sure folder exist 
            os.makedirs(inference.outputRTstructPatientStage1Temp, exist_ok=True)
            # Set output folder for fused RT structure data
            inference.outputRTstructPatientModelFused = os.path.join(base.workFolder, 'inference', inference.inferenceSubFolder, 'nnUNet_data', 'inference_RTstruct_Fused_out')
            # Make sure folder exist 
            os.makedirs(inference.outputRTstructPatientModelFused, exist_ok=True)

        # Return the configuration
        return base, preProcess, inference
        
