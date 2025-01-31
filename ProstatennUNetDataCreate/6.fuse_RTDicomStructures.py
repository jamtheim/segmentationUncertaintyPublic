# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Fusion of RT structures from DICOM format. 
# *********************************************************************************
# Test från Louise dator

import os
from joblib import Parallel, delayed
import multiprocessing
import subprocess
import numpy as np 
import time 
import matplotlib.pyplot as plt
import shutil
import dicom2nifti
from rt_utils import RTStructMerger # Copied folder to local storage from https://github.com/qurit/rt-utils. Do not pip install this! 
from rt_utils import RTStructBuilder
import psutil
from ioDataMethods import ioDataMethodsClass
from commonConfig import commonConfigClass
from convertDataMethods import convertDataMethodsClass

# Init needed class instances
ioData = ioDataMethodsClass()           # Class for handling and reading data 
conf = commonConfigClass()              # Init config class
convertData = convertDataMethodsClass() # Functions for converting DICOM to Nifti data

# Define models to use when fusing structures, loaded from config
fuseStructsFromTasks = conf.inference.fuseStructsFromTasks
# Loop over patients in inputNiftiPatientDir, use files in Nifti directory
patientNiftiSegFiles = os.listdir(conf.inference.inputNiftiPatientDir)
# Make sure it contain only Nifti files
patientNiftiSegFiles = [x for x in patientNiftiSegFiles if x.endswith(conf.preProcess.fileSuffix)]
# Split to patient name only
patientNames = [x.split('_')[0] for x in patientNiftiSegFiles]

# Define function for parallel processing
def patientLargeLoop(patient): 
    # Define an empty list for DicomRTStructFilePaths
    DicomRTStructFileAllPaths = []
    # Loop over task numbers
    for taskNumber in fuseStructsFromTasks: 
        # Define path for patient RT struct DICOM file 
        DicomRTStructFilePath = os.path.join(conf.inference.outputRTstructPatientDirBase + '_TaskNumber_' + str(taskNumber), str(patient) + '_RTstruct.dcm')
        DicomRTStructFileAllPaths.append(DicomRTStructFilePath)

    # Create path for patient DICOM CT files
    DICOMCTfolderPath = os.path.join(conf.inference.inputDicomPatientDir, str(patient), 'CT')
    # Fuse the first two RT DICOM strucures 
    mergedDICOMRTStructuresStage1 = RTStructMerger.merge_rtstructs(dicom_series_path=DICOMCTfolderPath, rt_struct_path1=DicomRTStructFileAllPaths[0], rt_struct_path2=DicomRTStructFileAllPaths[1])
    # Define path for temporary stage 1 DICOM RT STRUCT. We use this as we only can fuse two at a time. 
    tempPathStage1=os.path.join(conf.inference.outputRTstructPatientStage1Temp, str(patient) + '_merged-rt-struct_stage1.dcm')
    # Temp save 
    mergedDICOMRTStructuresStage1.save(tempPathStage1)
    # Merge of all 3 RT Structure Sets, temp is input. 
    mergedDICOMRTStructuresStage2 = RTStructMerger.merge_rtstructs(dicom_series_path=DICOMCTfolderPath, rt_struct_path1=tempPathStage1, rt_struct_path2=DicomRTStructFileAllPaths[2])
    # Define path for stage 2 DICOM RT STRUCT
    tempPathStage2 =  os.path.join(conf.inference.outputRTstructPatientModelFused, str(patient) + '_merged-rt-struct.dcm')
    # Save the merged final RT structure 
    mergedDICOMRTStructuresStage2.save(tempPathStage2)
    # Delete the temporary patient RT structure file 
    try: 
        os.remove(tempPathStage1)
    except:     
        pass
      
# Define number of CPU threads
nrCPU = multiprocessing.cpu_count()-2
# nrCPU = 1
# Excecute RT struct generation in parallel
Parallel(n_jobs=nrCPU, verbose=10)(delayed(patientLargeLoop)(patient) for patientNumber, patient in enumerate(patientNames))

# Assert that all files have been created, else give error 
fusedRTStructList = os.listdir(conf.inference.outputRTstructPatientModelFused)
fusedRTStructList = [x for x in fusedRTStructList if x.endswith('.dcm')]
assert len(fusedRTStructList) == len(patientNiftiSegFiles), 'Not all fused RT structure DICOM files have been created'

# Print message
print(' ')
print('Fusion of RT structures are complete')