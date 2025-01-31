# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: This script contains code for displaying the difference between reference and observer data.
# Output is histograms of the edited uncertainty voxels for the observer data as image files.
# This is also produces for aggregated data over all observers. 
# *********************************************************************************

import os
import pickle
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from commonConfig import commonConfigClass
from evalDataMethods import evalDataMethodsClass
from plotDataMethods import plotDataMethodsClass
from joblib import Parallel, delayed
import multiprocessing
# Load configuration
conf = commonConfigClass() # Configuration for the project
# Load methods
evalData = evalDataMethodsClass() # Functions for converting DICOM to Nifti data
# Load methods for plotting
plotData = plotDataMethodsClass() # Functions for plotting data

# Config 
nrCPU = conf.eval.nrCPU # Number of CPUs to use


def patLoopDisplay(observer, patient):
    """
    Loop through each patient and create uncertainty histograms for the observer data.
    The histograms are created for the CTV and Rectum structures for step 1 and 2.
    Also returns the selected uncertainty voxels for each step.

    Arg:
        observer (str): The observer to evaluate.
        patient (str): The patient to evaluate.

    Returns:
        dict: A dictionary containing the observer, patient and selected uncertainty voxels for each step.

    """
    # Print patient
    print(' ')
    print('Patient: ' + patient)
    # Load the uncertainty map for the patient (110 for CTV and 111 for Rectum)
    # CTV
    CTV_uncertaintyFilePath = os.path.join(conf.base.infDataBaseFolderPath, conf.base.infObsFolderName + '_' + observer, 'inference_NiftiSeg_out_TaskNumber_110', patient + '_' + observer + '_uncertaintyMap.nii.gz')
    CTV_uncertainty, CTV_uncertainty_sitk, CTV_uncertainty_pxlSpacing = evalData.readNiftiFile(CTV_uncertaintyFilePath, 'float32')
    #print('CTV uncertainty file path: ' + CTV_uncertaintyFilePath)
    # Rectum
    Rectum_uncertaintyFilePath = os.path.join(conf.base.infDataBaseFolderPath, conf.base.infObsFolderName + '_' + observer, 'inference_NiftiSeg_out_TaskNumber_111', patient + '_' + observer + '_uncertaintyMap.nii.gz')
    Rectum_uncertainty, Rectum_uncertainty_sitk, Rectum_uncertainty_pxlSpacing = evalData.readNiftiFile(Rectum_uncertaintyFilePath, 'float32')
    #print('Rectum uncertainty file path: ' + Rectum_uncertaintyFilePath)
    
    # Load the difference maps for the patient. Note that we are using the absolute difference maps here.
    # This gets us all (removed and added) voxels that are different between the reference and observer data.
    # CTV
    CTV_diff_step1 = evalData.loadNpzToMatrix(observer, patient, 'CTV', 1, 'absdiff')
    CTV_diff_step2 = evalData.loadNpzToMatrix(observer, patient, 'CTV', 2, 'absdiff')
    evalData.assertNiftiDataShape(CTV_uncertainty, CTV_diff_step1, 1, 1, 'CTV') # Pixel spacing is not used here
    evalData.assertNiftiDataShape(CTV_uncertainty, CTV_diff_step2, 1, 1, 'CTV') # Pixel spacing is not used here

    # Rectum 
    Rectum_diff_step1 = evalData.loadNpzToMatrix(observer, patient, 'Rectum', 1, 'absdiff')
    Rectum_diff_step2 = evalData.loadNpzToMatrix(observer, patient, 'Rectum', 2, 'absdiff')
    evalData.assertNiftiDataShape(Rectum_uncertainty, Rectum_diff_step1, 1, 1, 'Rectum') # Pixel spacing is not used here
    evalData.assertNiftiDataShape(Rectum_uncertainty, Rectum_diff_step2, 1, 1, 'Rectum') # Pixel spacing is not used here

    # Plot histograms and get number of voxels above threshold
    # CTV
    selUncVoxelCTVStep1, selUncVoxelCTVStep2, = plotData.plotHistogramFromMask(CTV_uncertainty, CTV_diff_step1, CTV_diff_step2, 
                                        ['Step1edits', 'Step2edits'], 'CTV', observer, patient)
    # Get amount of voxels above thresholds for step 1 and 2 and plot
    nrVoxelAboveThresholdCTVStep1 = plotData.getVoxelsAboveThreshold(CTV_uncertainty, CTV_diff_step1, 'Step1',
                                                                     'CTV', observer, patient, conf.eval.bin_edges)
    nrVoxelAboveThresholdCTVStep2 = plotData.getVoxelsAboveThreshold(CTV_uncertainty, CTV_diff_step2, 'Step2',
                                                                        'CTV', observer, patient, conf.eval.bin_edges)
   
    # Rectum
    selUncVoxelRectumStep1, selUncVoxelRectumStep2 = plotData.plotHistogramFromMask(Rectum_uncertainty, Rectum_diff_step1, Rectum_diff_step2, 
                                        ['Step1edits', 'Step2edits'], 'Rectum', observer, patient)
    # Get amount of voxels above threshold for step 1 and 2 and plot
    nrVoxelAboveThresholdRectumStep1 = plotData.getVoxelsAboveThreshold(Rectum_uncertainty, Rectum_diff_step1, 'Step1',
                                                                        'Rectum', observer, patient, conf.eval.bin_edges)
    nrVoxelAboveThresholdRectumStep2 = plotData.getVoxelsAboveThreshold(Rectum_uncertainty, Rectum_diff_step2, 'Step2',
                                                                        'Rectum', observer, patient, conf.eval.bin_edges)
    
    # Plot the two vectors for CTV and Rectum
    plotData.plotVoxelsAboveThresholds(nrVoxelAboveThresholdCTVStep1, nrVoxelAboveThresholdCTVStep2, ['Step1edits', 'Step2edits'], 'CTV', observer, patient)
    plotData.plotVoxelsAboveThresholds(nrVoxelAboveThresholdRectumStep1, nrVoxelAboveThresholdRectumStep2, ['Step1edits', 'Step2edits'], 'Rectum', observer, patient)
    
    
    # Return data in dictionary
    return {
            'observer': observer,
            'patient': patient,
            'selUncVoxelCTVStep1': selUncVoxelCTVStep1,
            'selUncVoxelCTVStep2': selUncVoxelCTVStep2,
            'selUncVoxelRectumStep1': selUncVoxelRectumStep1,
            'selUncVoxelRectumStep2': selUncVoxelRectumStep2,
            'nrVoxelAboveThresholdCTVStep1': nrVoxelAboveThresholdCTVStep1,
            'nrVoxelAboveThresholdCTVStep2': nrVoxelAboveThresholdCTVStep2,
            'nrVoxelAboveThresholdRectumStep1': nrVoxelAboveThresholdRectumStep1,
            'nrVoxelAboveThresholdRectumStep2': nrVoxelAboveThresholdRectumStep2
            }



### This is the entry point of the script ### 

# Init empty dictionary to collect the results
allResults = {}     
# Loop through each observer and calculate histogram data for each patient
for observer in conf.eval.observers:
    # Print observer
    print(' ')
    print('Observer: ' + observer)
    # Loop through each patient in parallel within the observer loop
    results = Parallel(n_jobs=nrCPU, verbose=10)(delayed(patLoopDisplay)(observer, patient) for patient in conf.eval.patients)
    # Initialize dictionary for the current observer
    allResults[observer] = {}
    # Organize results into the dictionary by patient
    for currResult in results:
        allResults[currResult['observer']][currResult['patient']] = { # currResult['observer'], this is how you access the observer key in the dictionary
            'selUncVoxelCTVStep1': currResult['selUncVoxelCTVStep1'],
            'selUncVoxelCTVStep2': currResult['selUncVoxelCTVStep2'],
            'selUncVoxelRectumStep1': currResult['selUncVoxelRectumStep1'],
            'selUncVoxelRectumStep2': currResult['selUncVoxelRectumStep2'],
            'nrVoxelAboveThresholdCTVStep1': currResult['nrVoxelAboveThresholdCTVStep1'],
            'nrVoxelAboveThresholdCTVStep2': currResult['nrVoxelAboveThresholdCTVStep2'],
            'nrVoxelAboveThresholdRectumStep1': currResult['nrVoxelAboveThresholdRectumStep1'],
            'nrVoxelAboveThresholdRectumStep2': currResult['nrVoxelAboveThresholdRectumStep2']
        }

# QA checks
# Check size of dictionary. Should be 35 patients for all observers. Loop over all observers
for observer in conf.eval.observers:
    assert len(allResults[observer]) == len(conf.eval.patients), 'The number of patients is not correct for observer ' + observer
# Check that number of observers is correct
assert len(allResults) == len(conf.eval.observers), 'The number of observers is not correct'

# Save histogram data to pickle file 
# Folder location is conf.base.histDataOutFolderPath, make sure it exists
if not os.path.exists(conf.base.histDataOutFolderPath):
    os.makedirs(conf.base.histDataOutFolderPath)
# Save the collected data to a file
with open(os.path.join(conf.base.histDataOutFolderPath, 'allUncVoxelResults.pkl'), 'wb') as f:
    pickle.dump(allResults, f)



### Aggregate voxel data for each patient over all observers ### 
# Allows visualization of all voxels in the cohort for a specific observer
# Create a dictionary for the aggregated data
aggDataPatient = {}
# Loop through each patient and calculate the aggregated uncertainty voxel data over all observers
for patient in conf.eval.patients:
    # Initialize dictionary for this patient
    aggDataPatient[patient] = {}
    # Initialize empty arrays for each structure
    aggDataPatient[patient]['CTVStep1'] = np.array([])
    aggDataPatient[patient]['CTVStep2'] = np.array([])
    aggDataPatient[patient]['RectumStep1'] = np.array([])
    aggDataPatient[patient]['RectumStep2'] = np.array([])
    aggDataPatient[patient]['nrVoxelAboveThresholdCTVStep1'] = np.array([])
    aggDataPatient[patient]['nrVoxelAboveThresholdCTVStep2'] = np.array([])
    aggDataPatient[patient]['nrVoxelAboveThresholdRectumStep1'] = np.array([])
    aggDataPatient[patient]['nrVoxelAboveThresholdRectumStep2'] = np.array([])
    # Loop through each observer
    for observer in conf.eval.observers:
        # Append the data for each observer
        aggDataPatient[patient]['CTVStep1'] = np.append(aggDataPatient[patient]['CTVStep1'], allResults[observer][patient]['selUncVoxelCTVStep1'])
        aggDataPatient[patient]['CTVStep2'] = np.append(aggDataPatient[patient]['CTVStep2'], allResults[observer][patient]['selUncVoxelCTVStep2'])
        aggDataPatient[patient]['RectumStep1'] = np.append(aggDataPatient[patient]['RectumStep1'], allResults[observer][patient]['selUncVoxelRectumStep1'])
        aggDataPatient[patient]['RectumStep2'] = np.append(aggDataPatient[patient]['RectumStep2'], allResults[observer][patient]['selUncVoxelRectumStep2'])
        aggDataPatient[patient]['nrVoxelAboveThresholdCTVStep1'] = np.append(aggDataPatient[patient]['nrVoxelAboveThresholdCTVStep1'], allResults[observer][patient]['nrVoxelAboveThresholdCTVStep1'])
        aggDataPatient[patient]['nrVoxelAboveThresholdCTVStep2'] = np.append(aggDataPatient[patient]['nrVoxelAboveThresholdCTVStep2'], allResults[observer][patient]['nrVoxelAboveThresholdCTVStep2'])
        aggDataPatient[patient]['nrVoxelAboveThresholdRectumStep1'] = np.append(aggDataPatient[patient]['nrVoxelAboveThresholdRectumStep1'], allResults[observer][patient]['nrVoxelAboveThresholdRectumStep1'])
        aggDataPatient[patient]['nrVoxelAboveThresholdRectumStep2'] = np.append(aggDataPatient[patient]['nrVoxelAboveThresholdRectumStep2'], allResults[observer][patient]['nrVoxelAboveThresholdRectumStep2'])
    
    # Plot aggregated data per patient
    # CTV
    plotData.plotHistogramFromVoxels(aggDataPatient[patient]['CTVStep1'], aggDataPatient[patient]['CTVStep2'], ['Step1edits', 'Step2edits'], 'CTV', 'All', patient)
    # Rectum
    plotData.plotHistogramFromVoxels(aggDataPatient[patient]['RectumStep1'], aggDataPatient[patient]['RectumStep2'], ['Step1edits', 'Step2edits'], 'Rectum', 'All', patient)
    # Sum and plot the two vectors for CTV and Rectum
    plotData.plotVoxelsAboveThresholds(evalData.sumDictColumns(aggDataPatient[patient]['nrVoxelAboveThresholdCTVStep1']), evalData.sumDictColumns(aggDataPatient[patient]['nrVoxelAboveThresholdCTVStep2']), ['Step1edits', 'Step2edits'], 'CTV', 'All', patient)
    plotData.plotVoxelsAboveThresholds(evalData.sumDictColumns(aggDataPatient[patient]['nrVoxelAboveThresholdRectumStep1']), evalData.sumDictColumns(aggDataPatient[patient]['nrVoxelAboveThresholdRectumStep2']), ['Step1edits', 'Step2edits'], 'Rectum', 'All', patient)
  


### Aggregate voxel data for all patients over each observer  ###
# Create a dictionary for the aggregated data
aggDataObserver = {}
# Loop through each observer and calculate the aggregated uncertainty voxel data over all patients
for observer in conf.eval.observers:
    # Initialize dictionary for this observer
    aggDataObserver[observer] = {}
    # Initialize empty arrays for each structure
    aggDataObserver[observer]['CTVStep1'] = np.array([])
    aggDataObserver[observer]['CTVStep2'] = np.array([])
    aggDataObserver[observer]['RectumStep1'] = np.array([])
    aggDataObserver[observer]['RectumStep2'] = np.array([])
    aggDataObserver[observer]['nrVoxelAboveThresholdCTVStep1'] = np.array([])
    aggDataObserver[observer]['nrVoxelAboveThresholdCTVStep2'] = np.array([])
    aggDataObserver[observer]['nrVoxelAboveThresholdRectumStep1'] = np.array([])
    aggDataObserver[observer]['nrVoxelAboveThresholdRectumStep2'] = np.array([])
    # Loop through each patient
    for patient in conf.eval.patients:
        # Append the data for each patient
        aggDataObserver[observer]['CTVStep1'] = np.append(aggDataObserver[observer]['CTVStep1'], allResults[observer][patient]['selUncVoxelCTVStep1'])
        aggDataObserver[observer]['CTVStep2'] = np.append(aggDataObserver[observer]['CTVStep2'], allResults[observer][patient]['selUncVoxelCTVStep2'])
        aggDataObserver[observer]['RectumStep1'] = np.append(aggDataObserver[observer]['RectumStep1'], allResults[observer][patient]['selUncVoxelRectumStep1'])
        aggDataObserver[observer]['RectumStep2'] = np.append(aggDataObserver[observer]['RectumStep2'], allResults[observer][patient]['selUncVoxelRectumStep2'])
        aggDataObserver[observer]['nrVoxelAboveThresholdCTVStep1'] = np.append(aggDataObserver[observer]['nrVoxelAboveThresholdCTVStep1'], allResults[observer][patient]['nrVoxelAboveThresholdCTVStep1'])
        aggDataObserver[observer]['nrVoxelAboveThresholdCTVStep2'] = np.append(aggDataObserver[observer]['nrVoxelAboveThresholdCTVStep2'], allResults[observer][patient]['nrVoxelAboveThresholdCTVStep2'])
        aggDataObserver[observer]['nrVoxelAboveThresholdRectumStep1'] = np.append(aggDataObserver[observer]['nrVoxelAboveThresholdRectumStep1'], allResults[observer][patient]['nrVoxelAboveThresholdRectumStep1'])
        aggDataObserver[observer]['nrVoxelAboveThresholdRectumStep2'] = np.append(aggDataObserver[observer]['nrVoxelAboveThresholdRectumStep2'], allResults[observer][patient]['nrVoxelAboveThresholdRectumStep2'])
    
    # Plot aggregated data over all patients per observer
    # CTV
    plotData.plotHistogramFromVoxels(aggDataObserver[observer]['CTVStep1'], aggDataObserver[observer]['CTVStep2'], ['Step1edits', 'Step2edits'], 'CTV', observer, 'All')
    # Rectum
    plotData.plotHistogramFromVoxels(aggDataObserver[observer]['RectumStep1'], aggDataObserver[observer]['RectumStep2'], ['Step1edits', 'Step2edits'], 'Rectum', observer, 'All')
    # Sum and plot the two vectors for CTV and Rectum
    plotData.plotVoxelsAboveThresholds(evalData.sumDictColumns(aggDataObserver[observer]['nrVoxelAboveThresholdCTVStep1']), evalData.sumDictColumns(aggDataObserver[observer]['nrVoxelAboveThresholdCTVStep2']), ['Step1', 'Step2'], 'CTV', observer, 'All')
    plotData.plotVoxelsAboveThresholds(evalData.sumDictColumns(aggDataObserver[observer]['nrVoxelAboveThresholdRectumStep1']), evalData.sumDictColumns(aggDataObserver[observer]['nrVoxelAboveThresholdRectumStep2']), ['Step1', 'Step2'], 'Rectum', observer, 'All')



### Aggregate metrics plot data for all patients over each observer (per observer) ###
# Create a dictionary to separately save data for each observer 
aggMetricAllObservers = {}

for observer in conf.eval.observers:
    # Step 1 vs ref
    segStep1vsRef_CTV, subjectList, files = plotData.aggregateCSVdata(observer, 'CTV', 'step1', 'segMetricsVsRef')
    segStep1vsRef_Rectum, subjectList, files = plotData.aggregateCSVdata(observer, 'Rectum', 'step1', 'segMetricsVsRef')
    # Store in dictionary
    aggMetricAllObservers[observer] = {
        'segStep1vsRef_CTV': segStep1vsRef_CTV,
        'segStep1vsRef_Rectum': segStep1vsRef_Rectum
    }
    # Step 2 vs ref
    segStep2vsRef_CTV, subjectList, files = plotData.aggregateCSVdata(observer, 'CTV', 'step2', 'segMetricsVsRef')
    segStep2vsRef_Rectum, subjectList, files = plotData.aggregateCSVdata(observer, 'Rectum', 'step2', 'segMetricsVsRef')
    # Store in dictionary
    aggMetricAllObservers[observer].update({
        'segStep2vsRef_CTV': segStep2vsRef_CTV,
        'segStep2vsRef_Rectum': segStep2vsRef_Rectum
    })
    # Step 1 vs STAPLE
    segStep1vsSTAPLE_CTV, subjectList, files = plotData.aggregateCSVdata(observer, 'CTV', 'step1', 'segMetricsVsSTAPLE')
    segStep1vsSTAPLE_Rectum, subjectList, files = plotData.aggregateCSVdata(observer, 'Rectum', 'step1', 'segMetricsVsSTAPLE')
    # Store in dictionary
    aggMetricAllObservers[observer].update({
        'segStep1vsSTAPLE_CTV': segStep1vsSTAPLE_CTV,
        'segStep1vsSTAPLE_Rectum': segStep1vsSTAPLE_Rectum
    })
    # Step 2 vs STAPLE
    segStep2vsSTAPLE_CTV, subjectList, files = plotData.aggregateCSVdata(observer, 'CTV', 'step2', 'segMetricsVsSTAPLE')
    segStep2vsSTAPLE_Rectum, subjectList, files = plotData.aggregateCSVdata(observer, 'Rectum', 'step2', 'segMetricsVsSTAPLE')
    # Store in dictionary
    aggMetricAllObservers[observer].update({
        'segStep2vsSTAPLE_CTV': segStep2vsSTAPLE_CTV,
        'segStep2vsSTAPLE_Rectum': segStep2vsSTAPLE_Rectum
    })
    # Difference step 1 vs step 2
    segStep1vsStep2_CTV, subjectList, files = plotData.aggregateCSVdata(observer, 'CTV', 'step12', 'segMetricsDiffStep12')
    segStep1vsStep2_Rectum, subjectList, files = plotData.aggregateCSVdata(observer, 'Rectum', 'step12', 'segMetricsDiffStep12')
    # Store in dictionary
    aggMetricAllObservers[observer].update({
        'segStep1vsStep2_CTV': segStep1vsStep2_CTV,
        'segStep1vsStep2_Rectum': segStep1vsStep2_Rectum
    })
    
    # Plots for each observer 
    # Create plots for different metrics (including statistical tests if needed)
    # Step 1 vs ref
    plotData.plotDictMetricPerObserver(observer, segStep1vsRef_CTV, 'segStep1vsRef_CTV') # Replicate variable name to second input as string, very important for resulting file name 
    plotData.plotDictMetricPerObserver(observer, segStep1vsRef_Rectum, 'segStep1vsRef_Rectum')
    # Step 2 vs ref
    plotData.plotDictMetricPerObserver(observer,segStep2vsRef_CTV, 'segStep2vsRef_CTV')
    plotData.plotDictMetricPerObserver(observer, segStep2vsRef_Rectum, 'segStep2vsRef_Rectum')
    # Step 1 vs STAPLE
    plotData.plotDictMetricPerObserver(observer, segStep1vsSTAPLE_CTV, 'segStep1vsSTAPLE_CTV')
    plotData.plotDictMetricPerObserver(observer, segStep1vsSTAPLE_Rectum, 'segStep1vsSTAPLE_Rectum')
    # Step 2 vs STAPLE
    plotData.plotDictMetricPerObserver(observer, segStep2vsSTAPLE_CTV, 'segStep2vsSTAPLE_CTV')
    plotData.plotDictMetricPerObserver(observer, segStep2vsSTAPLE_Rectum, 'segStep2vsSTAPLE_Rectum')
    # Difference step 1 vs step 2
    plotData.plotDictMetricPerObserver(observer, segStep1vsStep2_CTV, 'segStep1vsStep2_CTV')
    plotData.plotDictMetricPerObserver(observer, segStep1vsStep2_Rectum, 'segStep1vsStep2_Rectum')


# Plot aggregated metric data for all observers in the same plot
# Step 1 vs ref
plotData.plotDictMetricAllObservers(aggMetricAllObservers, 'segStep1vsRef_CTV')
plotData.plotDictMetricAllObservers(aggMetricAllObservers, 'segStep1vsRef_Rectum')
# Step 2 vs ref
plotData.plotDictMetricAllObservers(aggMetricAllObservers, 'segStep2vsRef_CTV')
plotData.plotDictMetricAllObservers(aggMetricAllObservers, 'segStep2vsRef_Rectum')
# Step 1 vs STAPLE
plotData.plotDictMetricAllObservers(aggMetricAllObservers, 'segStep1vsSTAPLE_CTV')
plotData.plotDictMetricAllObservers(aggMetricAllObservers, 'segStep1vsSTAPLE_Rectum')
# Step 2 vs STAPLE
plotData.plotDictMetricAllObservers(aggMetricAllObservers, 'segStep2vsSTAPLE_CTV')
plotData.plotDictMetricAllObservers(aggMetricAllObservers, 'segStep2vsSTAPLE_Rectum')
# Difference step 1 vs step 2
plotData.plotDictMetricAllObservers(aggMetricAllObservers, 'segStep1vsStep2_CTV')
plotData.plotDictMetricAllObservers(aggMetricAllObservers, 'segStep1vsStep2_Rectum')


