# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: This script contains code for calculating the difference between reference and observer data
# *********************************************************************************

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from commonConfig import commonConfigClass
from evalDataMethods import evalDataMethodsClass
from joblib import Parallel, delayed
import multiprocessing
# Load configuration
conf = commonConfigClass() # Configuration for the project
# Load methods
evalData = evalDataMethodsClass() # Functions for converting DICOM to Nifti data

# Config for parallell loop
nrCPU = conf.eval.nrCPU # Number of CPUs to use for parallell processing


def patDiffLoop(observer, patient):
    """
    Loop through each patient and calculate the difference between reference and observer data.
    Saves output to a file.
    Arg:
        observer (str): The current observer name
        patient (str): The current patient name
    """
    # Print patient
    print(' ')
    print('Patient: ' + patient)
    # Load the Nifti reference data for the patient 
    # CTV 
    CTV_niftiReferenceFilePath = os.path.join(conf.base.obsDataEditedFolderPath, observer, patient + '_' + observer, conf.base.referenceFolderName, 'CTV', conf.base.niftiFolderName, 'mask_' + patient + '_CTV_ref.nii.gz')
    CTV_ref, CTV_ref_sitk, CTV_ref_pxlSpacing = evalData.readNiftiFile(CTV_niftiReferenceFilePath, 'int8')
    evalData.assertBinaryInt8Data(CTV_ref)
    #evalData.saveMatrixToNpz(CTV_ref, observer, patient, 'CTV', 0, 'ref')
    # Rectum 
    Rectum_niftiReferenceFilePath = os.path.join(conf.base.obsDataEditedFolderPath, observer, patient + '_' + observer, conf.base.referenceFolderName, 'Rectum', conf.base.niftiFolderName, 'mask_' + patient + '_Rectum_ref.nii.gz')                  
    Rectum_ref, Rectum_ref_sitk, Rectum_ref_pxlSpacing = evalData.readNiftiFile(Rectum_niftiReferenceFilePath, 'int8')
    evalData.assertBinaryInt8Data(Rectum_ref)
    #evalData.saveMatrixToNpz(Rectum_ref, observer, patient, 'Rectum', 0, 'ref')

    # Load the Nifti observer data for the patient for step 1
    # CTV 
    CTV_step1, CTV_step1_sitk, CTV_step1_pxlSpacing = evalData.loadNiftiObserverData(observer, patient, 1, 'CTV', 'int8')
    evalData.assertNiftiDataShape(CTV_ref, CTV_step1, CTV_ref_pxlSpacing, CTV_step1_pxlSpacing, 'CTV') # Check that the data has the same shape
    evalData.assertBinaryInt8Data(CTV_step1)
    #evalData.saveMatrixToNpz(CTV_step1, observer, patient, 'CTV', 1, 'obs')
    # Rectum
    Rectum_step1, Rectum_step1_sitk, Rectum_step1_pxlSpacing = evalData.loadNiftiObserverData(observer, patient, 1, 'Rectum', 'int8')
    evalData.assertNiftiDataShape(Rectum_ref, Rectum_step1, Rectum_ref_pxlSpacing, Rectum_step1_pxlSpacing, 'Rectum') # Check that the data has the same shape
    evalData.assertBinaryInt8Data(Rectum_step1)
    #evalData.saveMatrixToNpz(Rectum_step1, observer, patient, 'Rectum', 1, 'obs')

    # Load the Nifti observer data for the patient for step 2
    # CTV 
    CTV_step2, CTV_step2_sitk, CTV_step2_pxlSpacing = evalData.loadNiftiObserverData(observer, patient, 2, 'CTV', 'int8')
    evalData.assertNiftiDataShape(CTV_ref, CTV_step2, CTV_ref_pxlSpacing, CTV_step2_pxlSpacing, 'CTV') 
    evalData.assertBinaryInt8Data(CTV_step2)
    #evalData.saveMatrixToNpz(CTV_step2, observer, patient, 'CTV', 2, 'obs')
    # Rectum
    Rectum_step2, Rectum_step2_sitk, Rectum_step2_pxlSpacing = evalData.loadNiftiObserverData(observer, patient, 2, 'Rectum', 'int8')
    evalData.assertNiftiDataShape(Rectum_ref, Rectum_step2, Rectum_ref_pxlSpacing, Rectum_step2_pxlSpacing, 'Rectum')
    evalData.assertBinaryInt8Data(Rectum_step2)
    #evalData.saveMatrixToNpz(Rectum_step2, observer, patient, 'Rectum', 2, 'obs')
    
    # FAKE DATA - FOR GENERATING FAKE DATA FOR TESTING
    if conf.debug.fakeAllSegData == True: 
        # Throw an error if the fake data flag is set to True, delete this stop if needed 
        raise ValueError('Fake data flag is set to True. Please set to False to use real data. This is to avoid confusion with fake data.')
        #### FAKE DATA - METHOD 2, CREATE STEP 1 AND STEP 2 DATA #### 
        CTV_step1 = evalData.fakeSegDataSelectPieSlice(CTV_ref, 2*np.pi*1/4)
        CTV_step2 = evalData.fakeSegDataSelectPieSlice(CTV_ref, 2*np.pi*2/4) # Change to for example 2/4
        evalData.assertBinaryInt8Data(CTV_step1)
        evalData.assertBinaryInt8Data(CTV_step2)
        Rectum_step1 = evalData.fakeSegDataSelectPieSlice(Rectum_ref, 2*np.pi*1/4)
        Rectum_step2 = evalData.fakeSegDataSelectPieSlice(Rectum_ref, 2*np.pi*2/4) # Change to for example 2/4
        evalData.assertBinaryInt8Data(Rectum_step1)
        evalData.assertBinaryInt8Data(Rectum_step2)

    # Load the STAPLE segmentation for the patient
    # Step 1
    # CTV
    CTV_step1_STAPLE, CTV_step1_STAPLE_sitk, CTV_step1_STAPLE_pxlSpacing = evalData.loadNiftiSTAPLEData('obsAll', patient, 1, 'CTV', 'int8')
    evalData.assertNiftiDataShape(CTV_ref, CTV_step1_STAPLE, CTV_ref_pxlSpacing, CTV_step1_STAPLE_pxlSpacing, 'CTV')
    evalData.assertBinaryInt8Data(CTV_step1_STAPLE)
    # Rectum
    Rectum_step1_STAPLE, Rectum_step1_STAPLE_sitk, Rectum_step1_STAPLE_pxlSpacing = evalData.loadNiftiSTAPLEData('obsAll', patient, 1, 'Rectum', 'int8')
    evalData.assertNiftiDataShape(Rectum_ref, Rectum_step1_STAPLE, Rectum_ref_pxlSpacing, Rectum_step1_STAPLE_pxlSpacing, 'Rectum')
    evalData.assertBinaryInt8Data(Rectum_step1_STAPLE)
    # Step 2
    # CTV
    CTV_step2_STAPLE, CTV_step2_STAPLE_sitk, CTV_step2_STAPLE_pxlSpacing = evalData.loadNiftiSTAPLEData('obsAll', patient, 2, 'CTV', 'int8')
    evalData.assertNiftiDataShape(CTV_ref, CTV_step2_STAPLE, CTV_ref_pxlSpacing, CTV_step2_STAPLE_pxlSpacing, 'CTV')
    evalData.assertBinaryInt8Data(CTV_step2_STAPLE)
    # Rectum
    Rectum_step2_STAPLE, Rectum_step2_STAPLE_sitk, Rectum_step2_STAPLE_pxlSpacing = evalData.loadNiftiSTAPLEData('obsAll', patient, 2, 'Rectum', 'int8')
    evalData.assertNiftiDataShape(Rectum_ref, Rectum_step2_STAPLE, Rectum_ref_pxlSpacing, Rectum_step2_STAPLE_pxlSpacing, 'Rectum')
    evalData.assertBinaryInt8Data(Rectum_step2_STAPLE)
    

    # Calculate the difference between some reference and observer data and save data to file as npz
    # This is performed in two versions, with sign and without (absolute difference) 
    # CTV
    CTV_diff_step1 = evalData.calcSegDifference(CTV_ref, CTV_step1, absFlag=False)
    CTV_absdiff_step1 = evalData.calcSegDifference(CTV_ref, CTV_step1, absFlag=True)
    evalData.saveMatrixToNpz(CTV_diff_step1, observer, patient, 'CTV', 1, 'diff')
    evalData.saveMatrixToNpz(CTV_absdiff_step1, observer, patient, 'CTV', 1, 'absdiff')
    CTV_diff_step2 = evalData.calcSegDifference(CTV_ref, CTV_step2, absFlag=False)
    CTV_absdiff_step2 = evalData.calcSegDifference(CTV_ref, CTV_step2, absFlag=True)
    evalData.saveMatrixToNpz(CTV_diff_step2, observer, patient, 'CTV', 2, 'diff')
    evalData.saveMatrixToNpz(CTV_absdiff_step2, observer, patient, 'CTV', 2, 'absdiff')
    # Save difference to Nifti file for visual inspection. Negative values are given the label value 2. 
    evalData.saveMatrixToNifti(evalData.convertDiffToNiftiLabels(CTV_diff_step1, 2), observer, patient, 'CTV', 1, 'diff')
    evalData.saveMatrixToNifti(evalData.convertDiffToNiftiLabels(CTV_diff_step2, 2), observer, patient, 'CTV', 2, 'diff')  

    # Rectum
    Rectum_diff_step1 = evalData.calcSegDifference(Rectum_ref, Rectum_step1, absFlag=False)
    Rectum_absdiff_step1 = evalData.calcSegDifference(Rectum_ref, Rectum_step1, absFlag=True)
    evalData.saveMatrixToNpz(Rectum_diff_step1, observer, patient, 'Rectum', 1, 'diff')
    evalData.saveMatrixToNpz(Rectum_absdiff_step1, observer, patient, 'Rectum', 1, 'absdiff')
    Rectum_diff_step2 = evalData.calcSegDifference(Rectum_ref, Rectum_step2, absFlag=False)
    Rectum_absdiff_step2 = evalData.calcSegDifference(Rectum_ref, Rectum_step2, absFlag=True)
    evalData.saveMatrixToNpz(Rectum_diff_step2, observer, patient, 'Rectum', 2, 'diff')
    evalData.saveMatrixToNpz(Rectum_absdiff_step2, observer, patient, 'Rectum', 2, 'absdiff')
    # Save difference to Nifti file for visual inspection. Negative values are given the label value 2.
    evalData.saveMatrixToNifti(evalData.convertDiffToNiftiLabels(Rectum_diff_step1, 2), observer, patient, 'Rectum', 1, 'diff')
    evalData.saveMatrixToNifti(evalData.convertDiffToNiftiLabels(Rectum_diff_step2, 2), observer, patient, 'Rectum', 2, 'diff')


    # Calculate metrics for the difference between some reference and observer data and STAPLE data 
    # This is based on Platipy and uses SITK objects as its input. 
    # CTV 
    CTV_step1_metrics = evalData.calcSegMetrics(CTV_ref_sitk, CTV_step1_sitk)
    evalData.saveDictToCSV(CTV_step1_metrics, observer, patient, 'CTV', 1, 'segMetricsVsRef')
    CTV_step1_STAPLE_metrics = evalData.calcSegMetrics(CTV_step1_STAPLE_sitk, CTV_step1_sitk)
    evalData.saveDictToCSV(CTV_step1_STAPLE_metrics, observer, patient, 'CTV', 1, 'segMetricsVsSTAPLE')
    CTV_step2_metrics = evalData.calcSegMetrics(CTV_ref_sitk, CTV_step2_sitk)
    evalData.saveDictToCSV(CTV_step2_metrics, observer, patient, 'CTV', 2, 'segMetricsVsRef')
    CTV_step2_STAPLE_metrics = evalData.calcSegMetrics(CTV_step2_STAPLE_sitk, CTV_step2_sitk)
    evalData.saveDictToCSV(CTV_step2_STAPLE_metrics, observer, patient, 'CTV', 2, 'segMetricsVsSTAPLE')
    # Rectum 
    Rectum_step1_metrics = evalData.calcSegMetrics(Rectum_ref_sitk, Rectum_step1_sitk)
    evalData.saveDictToCSV(Rectum_step1_metrics, observer, patient, 'Rectum', 1, 'segMetricsVsRef')
    Rectum_step1_STAPLE_metrics = evalData.calcSegMetrics(Rectum_step1_STAPLE_sitk, Rectum_step1_sitk)
    evalData.saveDictToCSV(Rectum_step1_STAPLE_metrics, observer, patient, 'Rectum', 1, 'segMetricsVsSTAPLE')
    Rectum_step2_metrics = evalData.calcSegMetrics(Rectum_ref_sitk, Rectum_step2_sitk)
    evalData.saveDictToCSV(Rectum_step2_metrics, observer, patient, 'Rectum', 2, 'segMetricsVsRef')
    Rectum_step2_STAPLE_metrics = evalData.calcSegMetrics(Rectum_step2_STAPLE_sitk, Rectum_step2_sitk)
    evalData.saveDictToCSV(Rectum_step2_STAPLE_metrics, observer, patient, 'Rectum', 2, 'segMetricsVsSTAPLE')
    # Calculate metrics for the difference between step 1 and step 2
    # This is based on Platipy and uses SITK objects.
    # CTV
    CTV_diffstep12_metrics = evalData.calcSegMetrics(CTV_step1_sitk, CTV_step2_sitk)
    evalData.saveDictToCSV(CTV_diffstep12_metrics, observer, patient, 'CTV', 12, 'segMetricsDiffStep12')
    # Rectum
    Rectum_diffstep12_metrics = evalData.calcSegMetrics(Rectum_step1_sitk, Rectum_step2_sitk)
    evalData.saveDictToCSV(Rectum_diffstep12_metrics, observer, patient, 'Rectum', 12, 'segMetricsDiffStep12')


def patSTAPLEandInterObsLoop(patient):
    """
    Loop through each patient and calculate the STAPLE segmentation for the observer data.
    Also calculate the inter-observer differences for the observer data as it used the same data as the STAPLE.
    Data is convertered to SITK format before calculation for STAPLE.
    """
    # Create a list to store the observer data for the patient
    observerStep1CTVList = []
    observerStep1RectumList = []
    observerStep2CTVList = []
    observerStep2RectumList = []

    # Loop over all observers for each patient and load the observer data
    for observer in conf.eval.observers:
        # Step 1
        # CTV
        CTV_step1, CTV_step1_sitk, CTV_step1_pxlSpacing = evalData.loadNiftiObserverData(observer, patient, 1, 'CTV', 'int8')
        evalData.assertBinaryInt8Data(CTV_step1)
        observerStep1CTVList.append(CTV_step1) 
        # Rectum
        Rectum_step1, Rectum_step1_sitk, Rectum_step1_pxlSpacing = evalData.loadNiftiObserverData(observer, patient, 1, 'Rectum', 'int8')
        evalData.assertBinaryInt8Data(Rectum_step1)
        observerStep1RectumList.append(Rectum_step1) 
        # Step 2
        # CTV
        CTV_step2, CTV_step2_sitk, CTV_step2_pxlSpacing = evalData.loadNiftiObserverData(observer, patient, 2, 'CTV', 'int8')
        evalData.assertBinaryInt8Data(CTV_step2)
        observerStep2CTVList.append(CTV_step2)
        # Rectum 
        Rectum_step2, Rectum_step2_sitk, Rectum_step2_pxlSpacing = evalData.loadNiftiObserverData(observer, patient, 2, 'Rectum', 'int8')
        evalData.assertBinaryInt8Data(Rectum_step2)
        observerStep2RectumList.append(Rectum_step2)
        # End of observer loop

    ##### Calculate the STAPLE segmentation for the observer data #####
    # Step 1
    # CTV
    observerStep1CTVList = np.stack(observerStep1CTVList, axis=-1)
    CTV_step1_STAPLE = evalData.calcSTAPLE(observerStep1CTVList, conf.eval.STAPLEThreshold)
    evalData.assertBinaryInt8Data(CTV_step1_STAPLE)
    evalData.saveMatrixToNpz(CTV_step1_STAPLE, 'obsAll', patient, 'CTV', 1, 'staple')
    evalData.saveMatrixToNifti(CTV_step1_STAPLE, 'obsAll', patient, 'CTV', 1, 'staple')
    # Rectum
    observerStep1RectumList = np.stack(observerStep1RectumList, axis=-1)
    Rectum_step1_STAPLE = evalData.calcSTAPLE(observerStep1RectumList, conf.eval.STAPLEThreshold)
    evalData.assertBinaryInt8Data(Rectum_step1_STAPLE)
    evalData.saveMatrixToNpz(Rectum_step1_STAPLE, 'obsAll', patient, 'Rectum', 1, 'staple')
    evalData.saveMatrixToNifti(Rectum_step1_STAPLE, 'obsAll', patient, 'Rectum', 1, 'staple')
    # Step 2
    # CTV
    observerStep2CTVList = np.stack(observerStep2CTVList, axis=-1)
    CTV_step2_STAPLE = evalData.calcSTAPLE(observerStep2CTVList, conf.eval.STAPLEThreshold)
    evalData.assertBinaryInt8Data(CTV_step2_STAPLE)
    evalData.saveMatrixToNpz(CTV_step2_STAPLE, 'obsAll', patient, 'CTV', 2, 'staple')
    evalData.saveMatrixToNifti(CTV_step2_STAPLE, 'obsAll', patient, 'CTV', 2, 'staple')
    # Rectum
    observerStep2RectumList = np.stack(observerStep2RectumList, axis=-1)
    Rectum_step2_STAPLE = evalData.calcSTAPLE(observerStep2RectumList, conf.eval.STAPLEThreshold)
    evalData.assertBinaryInt8Data(Rectum_step2_STAPLE)
    evalData.saveMatrixToNpz(Rectum_step2_STAPLE, 'obsAll', patient, 'Rectum', 2, 'staple')
    evalData.saveMatrixToNifti(Rectum_step2_STAPLE, 'obsAll', patient, 'Rectum', 2, 'staple')
    

    ##### Calculate the inter-observer differences for the observer data #####
    #### Voxel wise analasis of inter-observer differences was not used in the paper as it was deemed to unreliable ####
    # Collected data for each observer is stored from the needed data in
    # STAPLE calculation, specifically 
    # in the lists observerStep1CTVList, observerStep1RectumList, observerStep2CTVList, observerStep2RectumList

    # Define the reference structure to compare all observer structure to. To be able to used a paired test in the statistical analysis we will
    # use the AI reference as the reference for all observer data, as it is the same in step 1 and step 2. This is of cource patient specific.
    # In the file path below obsB is used but the reference is the same for all observers for each patient, so it does not matter. 
    observer = 'obsB'
    CTV_niftiReferenceFilePath = os.path.join(conf.base.obsDataEditedFolderPath, observer, patient + '_' + observer, conf.base.referenceFolderName, 'CTV', conf.base.niftiFolderName, 'mask_' + patient + '_CTV_ref.nii.gz')
    CTV_ref, CTV_ref_sitk, CTV_ref_pxlSpacing = evalData.readNiftiFile(CTV_niftiReferenceFilePath, 'int8')
    evalData.assertBinaryInt8Data(CTV_ref)
    # Rectum 
    Rectum_niftiReferenceFilePath = os.path.join(conf.base.obsDataEditedFolderPath, observer, patient + '_' + observer, conf.base.referenceFolderName, 'Rectum', conf.base.niftiFolderName, 'mask_' + patient + '_Rectum_ref.nii.gz')                  
    Rectum_ref, Rectum_ref_sitk, Rectum_ref_pxlSpacing = evalData.readNiftiFile(Rectum_niftiReferenceFilePath, 'int8')
    evalData.assertBinaryInt8Data(Rectum_ref)
    observer = [] # For safety
        
    # Create a function to calculate the aggregated statistics per vocel of signed distances between the observer data and the reference data
    def calcSurfaceDistancesSigned(observerData, reference, observers, pixelSpacing):
        """
        Calculate the signed distances between the observer data and the reference data.
        Args:
            observerData: The observer data from all observers
            reference: The reference data to compare with
            observers: The list of observers
            pixelSpacing: The pixel spacing of the data
        Returns:
            meanDistances: The mean signed distances for each voxel
            stdDistances: The std of the signed distances for each voxel
        """
        # Assert observerData to be numpy array
        assert isinstance(observerData, np.ndarray)
        # Assert observers and pixelSpacing to be lists
        assert isinstance(observers, list)
        # Assert that len observers equals last dim of observerList
        assert len(observers) == observerData.shape[-1]
        # Create an empty variable to store the surface distance matrix for each observer after calculatation of the signed distances
        allSurfaceDistances = np.zeros(reference.shape + (len(observers),))
        # Calculate signed distances for every voxel to the reference mask
        # Loop over the last dimension (observers) of the observerData and calculate the distance to the reference mask
        for i in range(observerData.shape[-1]):
            obsToRefDistance = evalData.surfaceDistancesSigned(observerData[:,:,:,i], reference, np.array(pixelSpacing))
            # Store the distance matrix in the allSurfaceDistances variable
            allSurfaceDistances[...,i] = obsToRefDistance
        # Calculate the mean and std of the signed distances for each voxel. Observer data is in the last dimension (-1)
        meanSurfaceDistances = np.mean(allSurfaceDistances, axis=-1)
        #stdSurfaceDistances = np.abs(allSurfaceDistances).std(axis=-1)
        # No absolute value is taken for the std, as we want to know the std of the signed distances
        stdSurfaceDistances = allSurfaceDistances.std(axis=-1)
        # Return data 
        return meanSurfaceDistances, stdSurfaceDistances
      

    # Step 1
    # CTV
    meanSurfaceDistancesCTVStep1, stdSurfaceDistancesCTVStep1 = calcSurfaceDistancesSigned(observerStep1CTVList, CTV_ref, conf.eval.observers, CTV_ref_pxlSpacing)
    evalData.saveMatrixToNpz(meanSurfaceDistancesCTVStep1, 'obsAll', patient, 'CTV', 1, 'meanSurfaceDistances')
    evalData.saveMatrixToNpz(stdSurfaceDistancesCTVStep1, 'obsAll', patient, 'CTV', 1, 'stdSurfaceDistances')
    print('CTV Step 1, Median of std values > 0: ' + str(np.median(stdSurfaceDistancesCTVStep1[stdSurfaceDistancesCTVStep1 > 0])))
    # Rectum
    meanSurfaceDistancesRectumStep1, stdSurfaceDistancesRectumStep1 = calcSurfaceDistancesSigned(observerStep1RectumList, Rectum_ref, conf.eval.observers, Rectum_ref_pxlSpacing)
    evalData.saveMatrixToNpz(meanSurfaceDistancesRectumStep1, 'obsAll', patient, 'Rectum', 1, 'meanSurfaceDistances')
    evalData.saveMatrixToNpz(stdSurfaceDistancesRectumStep1, 'obsAll', patient, 'Rectum', 1, 'stdSurfaceDistances')
    print('Rectum Step 1, Median of std values > 0: ' + str(np.median(stdSurfaceDistancesRectumStep1[stdSurfaceDistancesRectumStep1 > 0])))
    # Step 2
    # CTV
    meanSurfaceDistancesCTVStep2, stdSurfaceDistancesCTVStep2 = calcSurfaceDistancesSigned(observerStep2CTVList, CTV_ref, conf.eval.observers, CTV_ref_pxlSpacing)
    evalData.saveMatrixToNpz(meanSurfaceDistancesCTVStep2, 'obsAll', patient, 'CTV', 2, 'meanSurfaceDistances')
    evalData.saveMatrixToNpz(stdSurfaceDistancesCTVStep2, 'obsAll', patient, 'CTV', 2, 'stdSurfaceDistances')
    print('CTV Step 2, Median of std values > 0: ' + str(np.median(stdSurfaceDistancesCTVStep2[stdSurfaceDistancesCTVStep2 > 0])))
    # Rectum
    meanSurfaceDistancesRectumStep2, stdSurfaceDistancesRectumStep2 = calcSurfaceDistancesSigned(observerStep2RectumList, Rectum_ref, conf.eval.observers, Rectum_ref_pxlSpacing)
    evalData.saveMatrixToNpz(meanSurfaceDistancesRectumStep2, 'obsAll', patient, 'Rectum', 2, 'meanSurfaceDistances')
    evalData.saveMatrixToNpz(stdSurfaceDistancesRectumStep2, 'obsAll', patient, 'Rectum', 2, 'stdSurfaceDistances')
    print('Rectum Step 2, Median of std values > 0: ' + str(np.median(stdSurfaceDistancesRectumStep2[stdSurfaceDistancesRectumStep2 > 0])))
    # End of patient loop



# This is the entry point of the script

### Calculate STAPLE segmentations and inter-observer differences for the observer data ###
# STAPLE This needs to be calculated before the difference between reference and observer 
# data can be calculated in next parallell loop
# as we also calculate the difference between the STAPLE and observer data
# Loop over all patients in parallell and calculate the STAPLE segmentation over all observers
Parallel(n_jobs=int(nrCPU/4), verbose=10)(delayed(patSTAPLEandInterObsLoop)(patient) for patient in conf.eval.patients) # Surface distance calculation is memory intensive, use less CPUs

### Loop through each observer and calculate difference for each patient ###
for observer in conf.eval.observers:
    # Print observer
    print(' ')
    print('Observer: ' + observer)
    # Loop through each patient in parallel
    Parallel(n_jobs=nrCPU, verbose=10)(delayed(patDiffLoop)(observer, patient) for patient in conf.eval.patients)
