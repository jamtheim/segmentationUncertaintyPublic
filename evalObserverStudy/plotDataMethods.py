# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Class for plotting and describing data for the observer study
# *********************************************************************************

import numpy as np
import pandas as pd
import os
import math
import nibabel as nib
import SimpleITK as sitk  
from commonConfig import commonConfigClass
from evalDataMethods import evalDataMethodsClass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

# Load configuration
conf = commonConfigClass() 
# Load methods
evalData = evalDataMethodsClass()


class plotDataMethodsClass:
    """
    Class describing functions needed for evaluation of observer data
    """

    def __init__ (self):
        """
        Init function
        """
        pass


    def aggregateCSVdata(self, observer, organ, step, description):
        """
        Aggregates data from multiple patient CSV files into a dictionary based on the specified parameters given as input.
        Input parameters to dunction determines what files to analyse. 

        Args:
            observer (str): Observer to filter the files (e.g., "obsB").
            organ (str): Organ to filter the files (e.g., "CTV").
            step (str): Step to filter the files (e.g., "step2").
            description (str): Metric to filter the files (e.g., "segMetricsVsSTAPLE").

        Returns:
            dataDict: A dictionary with keys from all CSV files and values aggregated from all patients.
            subjects: A list of subjects.
        """
        # Init dictionary and list
        dataDict = {}
        subjects = []
        files = []
        
        # Loop over all specified patients
        for patient in conf.eval.patients:
            # Create filename 
            filename = f"{patient}_{observer}_{organ}_{step}_{description}.csv"
            # Create file path
            file_path = os.path.join(conf.base.evalDataEditedOutFolderPath, filename)
            # Check if file exists
            if os.path.isfile(file_path):
                # Append patient to subjects list
                subjects.append(patient)
                # Append file to files list
                files.append(filename)
                # Read the CSV file
                df = pd.read_csv(file_path, header=None, index_col=0)
                # Iterate through each key (index) in the DataFrame
                for key in df.index:
                    if key == 'APL': # Added path length for each slice
                        continue  # Skip processing 'APL' key
                    if key not in dataDict:
                        # Initialize a list for the key in dataDict
                        dataDict[key] = {'values': [], 'patients': []}
                    # Get values for the current key
                    values = df.loc[key].values
                    # Assert there is exactly one value per patient
                    assert len(values) == 1, f"Expected one value for key '{key}' for patient '{patient}'"
                    # Append the value to the corresponding key in dataDict
                    dataDict[key]['values'].append(values[0])
                    dataDict[key]['patients'].append(patient)
   
        # Make sure number of subject are the same as in conf.eval.patients
        # This checks that all patients have been processed and had a file generated
        assert len(subjects) == len(conf.eval.patients), 'Number of subjects does not match the number of patients'  
        assert len(files) == len(conf.eval.patients), 'Number of files does not match the number of patients'
        # Assert number of keys to be equal to number of metrics (conf.eval.nrGeometricMetrics)
        assert len(dataDict) == conf.eval.nrGeometricMetrics, 'Number of keys in dataDict does not match the number of geometric metrics'
        # Make sure every key in dictinary has the same number of values as subjects
        for key in dataDict:
            assert len(dataDict[key]['values']) == len(subjects), f"Number of values for key '{key}' does not match the number of subjects"      
        for key in dataDict:
            assert len(dataDict[key]['patients']) == len(subjects), f"Number of patients for key '{key}' does not match the number of subjects"      
        # Return the aggregated data dictionary and the list of subjects        
        return dataDict, subjects, files


    def plotDictMetricAllObservers(self, dataDict, comparison):
        """
        Plots the metric data for all keys from the aggregated data dictionary.
        dataDict contains data for all observers and all patients in the cohort.

        Args: 
            dataDict (dict): The dictionary with aggregated data.
            comparison (str): The name of the compariosn experiment.

        """
        # Asserts types
        assert isinstance(dataDict, dict)
        assert isinstance(comparison, str)
        assert len(dataDict) == len(conf.eval.observers), 'Number of observers in dataDict does not match the number of observers in the data'
        
        # For each observer in the data dictionary, QA the data and add to plot 
        for observer in dataDict:
            # Save list of metrics in a variable (will be overwritten in the loop, OK) 
            availableMetrics = dataDict[observer][comparison].keys()
            # Make sure every key in dictinary has the same number of values as subjects
            for key in dataDict[observer][comparison]:
                assert len(dataDict[observer][comparison][key]['values']) == len(conf.eval.patients), f"Number of values for key '{key}' does not match the number of subjects"      
            # Assert that patients are in the data dictionary and that the number of patients is correct
            for key in dataDict[observer][comparison]:
                assert 'patients' in dataDict[observer][comparison][key], f"Key '{key}' does not contain 'patients' key" 
            for key in dataDict[observer][comparison]:
                assert len(dataDict[observer][comparison][key]['patients']) == len(conf.eval.patients), f"Number of patients for key '{key}' does not match the number of subjects"      
        

        # Loop over all keys available in the data dictionary and create a plot for each key
        # where the data from each observer is overlayed in the same plot
        for key in availableMetrics:
            print(f"Plotting data for all observers for {comparison} and key {key}")
            # Create a new figure 
            plt.figure(figsize=conf.eval.figSize) 
            
            # Loop over each observer to add plot data 
            for observer in dataDict:
                # Create sns scatterplot
                sns.scatterplot(x=range(len(dataDict[observer][comparison][key]['patients'])), 
                                y=list(map(float, dataDict[observer][comparison][key]['values'])), 
                                s=conf.eval.plotSizeMetricAllObservers, label=observer, color=conf.eval.colorsMetricAllObservers[observer], marker=conf.eval.markersMetricAllObservers[observer])
                # Set x-ticks to have a tick for every data point
                xtick_labels = [patient.rsplit('_', 1)[-1] for patient in dataDict[observer][comparison][key]['patients']]
                plt.xticks(ticks=range(len(dataDict[observer][comparison][key]['values'])), labels=xtick_labels, rotation=90)
                plt.title(f"{comparison}_{key}")
                plt.xlabel("Subject")
                plt.ylabel(key + f" ({conf.eval.unitDict[key]})")
                plt.grid(True)
            
            # Add legend
            plt.legend(loc='upper right')
            
            # Save the plot as a PNG file, with the patient, structure, and observers (all) in the filename
            if not os.path.exists(conf.base.plotDataOutFolderPath):
                os.makedirs(conf.base.plotDataOutFolderPath)
            # Save the plot as a PNG file, with comparison and key in the filename
            plt.savefig(os.path.join(conf.base.plotDataOutFolderPath, f'obsAll_{comparison}_{key}.png'))
            # Close figure
            plt.close()
    

    def plotDictMetricPerObserver(self, observer, dataDict, dataDictName):
        """
        Plots the data for all keys from the aggregated data dictionary.
        dataDict is aggregated over all subjects in the cohort. 

        Args:
            observer (str): The observer to plot data for.
            data_dict (dict): The dictionary with aggregated data.
            dataDictName (str): The name of the data dictionary.
        """
        # Asserts types
        assert isinstance(dataDict, dict)
        assert isinstance(dataDictName, str)
        assert len(dataDict) == conf.eval.nrGeometricMetrics, 'Number of keys in dataDict does not match the number of geometric metrics'
        # Make sure every key in dictinary has the same number of values as subjects
        for key in dataDict:
            assert len(dataDict[key]['values']) == len(conf.eval.patients), f"Number of values for key '{key}' does not match the number of subjects"      
        # Assert that patients are in the data dictionary and that the number of patients is correct
        for key in dataDict:
            assert 'patients' in dataDict[key], f"Key '{key}' does not contain 'patients' key" 
        for key in dataDict:
            assert len(dataDict[key]['patients']) == len(conf.eval.patients), f"Number of patients for key '{key}' does not match the number of subjects"      
           
        # Loop over all keys available in the data dictionary and create a plot for each key
        for key in dataDict:
            print(f"Plotting data for " + observer + " and key " + key + " for " + dataDictName)
            # Create a new figure 
            plt.figure(figsize=conf.eval.figSize) 
            # Create sns scatterplot
            sns.scatterplot(x=range(len(dataDict[key]['patients'])), y=list(map(float, dataDict[key]['values'])), color='red', marker='o', s=50)
            # Set x-ticks to have a tick for every data point
            xtick_labels = [patient.rsplit('_', 1)[-1] for patient in dataDict[key]['patients']]
            plt.xticks(ticks=range(len(dataDict[key]['values'])), labels=xtick_labels, rotation=90)
            plt.title(f"{dataDictName}_{key}")
            plt.xlabel("Subject")
            plt.ylabel(key + f" ({conf.eval.unitDict[key]})")
            plt.grid(True)
            # Insert the mean and median to the upper right corner of the plot
            mean = np.mean(list(map(float, dataDict[key]['values'])))
            median = np.median(list(map(float, dataDict[key]['values'])))
            std = np.std(list(map(float, dataDict[key]['values'])))
            # Min and max
            minVal = np.min(list(map(float, dataDict[key]['values'])))
            maxVal = np.max(list(map(float, dataDict[key]['values'])))
            #plt.text(0.85, 0.98, f"Mean: {mean:.2f}\nMedian: {median:.2f}\nStd: {std:.2f}", horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, color='blue', fontsize=10)
            plt.text(0.85, 0.98, f"Mean: {mean:.2f}\nMedian: {median:.2f}\nStd: {std:.2f}\nMin: {minVal:.2f}\nMax: {maxVal:.2f}", horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, color='blue', fontsize=10)
            # Do a wilcoxon signed-rank test for certain keys if all data is not zero or all ones ()
            if key in conf.eval.keysWilcoxonTest:
                if not all(value == 0 for value in list(map(float, dataDict[key]['values']))) and not all(value == 1 for value in list(map(float, dataDict[key]['values']))): 
                    # Sample differences
                    differences = list(map(float, dataDict[key]['values']))
                    # If Key is VolumeRatio transform to log scale
                    if key == 'VolumeRatio':
                        differences = np.log(differences) 
                    # Perform the Wilcoxon signed-rank test
                    statistic, p_value = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
                    # if p-value is less than 0.05, print significat or not with large red text
                    if p_value < conf.eval.referencePvalue :
                        plt.text(0.75, 0.1, f"*Significant", horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, color='green', fontsize=20)
                        # Write p-value in the plot
                        plt.text(0.75, 0.05, f"p-value: {p_value:.2e}", horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, color='black', fontsize=10)
                    else: 
                        plt.text(0.75, 0.1, f"*Not significant", horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes, color='red', fontsize=20)

            # Save the plot as a PNG file, with the patient, structure, and observer in the filename
            if not os.path.exists(conf.base.plotDataOutFolderPath):
                os.makedirs(conf.base.plotDataOutFolderPath)
            # Save the plot as a PNG file, with dataDictName and key in the filename
            plt.savefig(os.path.join(conf.base.plotDataOutFolderPath, f'{observer}_{dataDictName}_{key}.png'))
            # Close figure
            plt.close()


    def getVoxelsAboveThreshold(self, uncertaintyMap, binaryMask, label, structure, observer, patient, thresholds):
        """
        Get the amount of voxels above a certain threshold in the uncertainty map.
        Limited by the binary mask.
        
        """
        # Get uncertainty map voxels contained within the binary mask
        selectedUncVoxels = self.selectVoxels(uncertaintyMap, binaryMask, structure, observer, patient)
        # Create a dictionary to store the amount of voxels above the thresholds
        voxelsAboveThreshold = {}
        # Get the amount of voxels above the thresholds, loop through thresholds
        for threshold in thresholds:
            # Count amount of voxels above the threshold
            count = np.sum(selectedUncVoxels > threshold)
            # Store the amount of voxels above the threshold in the dictionary
            voxelsAboveThreshold[str(threshold)] = count
        # Assert same number of values as bin edges
        assert len(voxelsAboveThreshold) == len(conf.eval.bin_edges), 'Number of values in voxelsAboveThreshold does not match the number of bin edges' 
        # Return the dictionary with the amount of voxels above the thresholds
        return voxelsAboveThreshold
    

    def plotVoxelsAboveThresholds(self, vector1, vector2, labels, structure, observer, patient):
        """
        Plot the amount of edited voxels above thresholds for two sets of voxels as a function of threshold.
        Args:
            vector1 (dict): The first set of voxels with thresholds as keys and counts as values.
            vector2 (dict): The second set of voxels with thresholds as keys and counts as values.
            labels (list): A list containing the labels for the steps.
            structure (str): The structure to evaluate.
            observer (str): The observer to evaluate.
            patient (str): The patient to evaluate.
        """
        # Assert vector1 and vector2 have the same length
        assert len(vector1) == len(vector2), 'The vectors do not have the same length'
        # Create a new figure 
        plt.figure(figsize=conf.eval.figSize) 
        # Convert dictionaries to lists for scatter plot
        x1, y1 = zip(*sorted(vector1.items()))
        x2, y2 = zip(*sorted(vector2.items()))
        # Create a scatter plot with different symbols
        sns.scatterplot(x=x1, y=y1, label=labels[0], color='blue', marker='s')  # Squares
        sns.scatterplot(x=x2, y=y2, label=labels[1], color='red', marker='*')   # Stars
        # Add labels and title
        plt.xlabel('Threshold')
        plt.ylabel('Voxel count')
        plt.title(f'Voxels above threshold for {structure} for observer(s) {observer} for patient {patient}')
        plt.legend(loc='upper right')
        # Set max y value from dictionary in conf.eval.plotVoxelsAboveThresholdsMaxYvalue
        # plt.ylim(0, conf.eval.plotVoxelsAboveThresholdsMaxYvalue[patient])
        # Save the plot as a PNG file, with the patient, structure, and observer in the filename
        if not os.path.exists(conf.base.histDataOutFolderPath):
            os.makedirs(conf.base.histDataOutFolderPath)
        # Save the plot as a PNG file, with the patient, structure and observer in the filename
        plt.savefig(os.path.join(conf.base.histDataOutFolderPath, f'{patient}_{structure}_voxelsAboveThreshold_{observer}.png'))
        

    def selectVoxels(self, uncertaintyMap, binaryMask, structure, observer, patient):
        """
        Select and flatten voxels from uncertaintyMap based on binaryMask.
        Args:
            uncertaintyMap (np.array): The uncertainty map to select voxels from.
            binaryMask (np.array): The binary mask with selected voxels.
        """
        # Assert binary mask
        evalData.assertBinaryInt8Data(binaryMask)
        # Select voxels in uncertaintyMap where corresponding voxel in binaryMask is 1 and flatten the array
        selectedUncVoxels = uncertaintyMap[binaryMask == 1].flatten()
        # Return selected voxels for further analysis
        return selectedUncVoxels
    
    
    def plotHistogramFromMask(self, uncertaintyMap, binaryMask1, binaryMask2, labels, structure, observer, patient):
        """
        Plot a histogram from where voxels are selected based on binary masks.
        Args:
            uncertaintyMap (np.array): The uncertainty map to select voxels from.
            binaryMask1 (np.array): The binary mask with selected voxels for the first step.
            binaryMask2 (np.array): The binary mask with selected voxels for the second step.
            labels (list): A list containing the labels for the steps.
            structure (str): The structure to evaluate.
            observer (str): The observer to evaluate.
            patient (str): The patient to evaluate.
        """
        # Select voxels in uncertaintyMap where corresponding voxel in binaryMask is 1 and flatten the array
        selectedUncVoxels1 = self.selectVoxels(uncertaintyMap, binaryMask1, structure, observer, patient)
        selectedUncVoxels2 = self.selectVoxels(uncertaintyMap, binaryMask2, structure, observer, patient)
        # Plot the histogram
        self.plotHist(selectedUncVoxels1, selectedUncVoxels2, labels, structure, observer, patient, uncertaintyMap)
        # Return selected voxels for further analysis
        return selectedUncVoxels1, selectedUncVoxels2

        
    def plotHistogramFromVoxels(self, selectedUncVoxels1, selectedUncVoxels2, labels, structure, observer, patient):
        """
        Plot a histogram from two sets of selected voxels.
        Args:
            selectedUncVoxels1 (np.array): The selected voxels for the first step.
            selectedUncVoxels2 (np.array): The selected voxels for the second step.
            labels (list): A list containing the labels for the steps.
            structure (str): The structure to evaluate.
            observer (str): The observer to evaluate.
            patient (str): The patient to evaluate.
        """
        self.plotHist(selectedUncVoxels1, selectedUncVoxels2, labels, structure, observer, patient)


    def plotHist(self, selectedUncVoxels1, selectedUncVoxels2, labels, structure, observer, patient, wholePatientUncertaintyMap=None):
        """
        Plot a histogram from two sets of selected voxels. 

        Args:
            wholeUncertaintyMap (np.array): The whole uncertainty map. Used to calculate total number of voxels for each bin.
            selectedUncVoxels1 (np.array): The selected voxels for the first step.
            selectedUncVoxels2 (np.array): The selected voxels for the second step.
            labels (list): A list containing the labels for the steps.
            structure (str): The structure to evaluate.
            observer (str): The observer to evaluate.
            patient (str): The patient to evaluate.
        """
        # Create a new figure 
        plt.figure(figsize=conf.eval.figSize) 
        # Set bin edges
        bin_edges = conf.eval.bin_edges
        # Assert that no value is outside the bin edges
        assert np.all(selectedUncVoxels1 <= bin_edges[-1]), 'Some values are outside the bin edges for binary mask 1'
        assert np.all(selectedUncVoxels2 <= bin_edges[-1]), 'Some values are outside the bin edges for binary mask 2'
        # Bin the data
        selectedVoxels1_binned = np.histogram(selectedUncVoxels1, bins=bin_edges)[0]
        selectedVoxels2_binned = np.histogram(selectedUncVoxels2, bins=bin_edges)[0]
        # Create a DataFrame for easier plotting with Seaborn
        bins = bin_edges[:-1]
        df = pd.DataFrame({
            'Bin': bins,
            labels[0]: selectedVoxels1_binned,
            labels[1]: selectedVoxels2_binned
        })
        # Melt the DataFrame to long format for Seaborn
        df_melted = df.melt(id_vars='Bin', value_vars=[labels[0], labels[1]], var_name='ObserverStudy', value_name='Count')
        # Create the bar plot with black edges
        ax = sns.barplot(x='Bin', y='Count', hue='ObserverStudy', data=df_melted, palette=['blue', 'red'], edgecolor='black')
        # Add labels and title
        plt.xlabel('Uncertainty map bins', fontsize=18)
        plt.ylabel('Voxel frequency count', fontsize=18)
        plt.title(f'{structure} for observer(s) {observer} for patient {patient}', fontsize=20)
        plt.xticks(ticks=np.arange(len(bins)), labels=[f'{b:.2f}-{b + 0.1:.2f}' for b in bins], fontsize=18)
        # Set font size for y ticks
        plt.yticks(fontsize=18)
        plt.legend(loc='upper right', fontsize=18)
        # Set ylim
        # Get current y-axis limits
        ymin, ymax = plt.ylim()
        # Add a buffer of 10% so accomoate the annotation s
        plt.ylim(ymin, ymax*1.1)
        # if observer is NOT 'All', set max y value from dictionary in conf.eval.plotHistogramFromMaskMaxYvalue
        if observer != 'All':
            pass
            #plt.ylim(0, conf.eval.plotHistogramFromMaskMaxYvalue[patient])

        # Set text above every bar in the bar plot with the number of selected voxels, the total number of voxels, and the percentage of selected voxels
        # This is not the prettiest code as I had to hardcode positions of bars in the plot. Seaborn does not provide a good way to get the x index of the bar in the plot. 
        # Frustrating to say the least...
        if wholePatientUncertaintyMap is not None:
            # We should only look with in the structure. So create the union of step 1 and step 2 masks.
            # This enables fair comparison of the percentage of changed voxels between step 1 and step 2. 
            mask_step1, _, _ = evalData.loadNiftiObserverData(observer, patient, 1, structure, 'int8') 
            mask_step2, _, _ = evalData.loadNiftiObserverData(observer, patient, 2, structure, 'int8')
            # Load reference structure mask
            reference_niftiReferenceFilePath = os.path.join(conf.base.obsDataEditedFolderPath, observer, patient + '_' + observer, conf.base.referenceFolderName, structure, conf.base.niftiFolderName, 'mask_' + patient + '_' + structure + '_ref.nii.gz')
            mask_ref, _, _ = evalData.readNiftiFile(reference_niftiReferenceFilePath, 'int8')
            # Union of all masks
            union_mask_steps = np.bitwise_or(mask_step1, mask_step2)
            union_mask = np.bitwise_or(union_mask_steps, mask_ref)
            # Get the wholePatientUncertaintyMap for the structure union voxels only 
            wholePatientUncertaintyMap = wholePatientUncertaintyMap[union_mask == 1].flatten()
            # Bin data from the truncated whole uncertainty map to get the total number of voxels available (not edited) for each bin
            wholePatientUncertaintyMap_binned = np.histogram(wholePatientUncertaintyMap, bins=bin_edges)[0]
            # Get number of datasets (step 1 and step 2)
            nrDatasets = len(labels)           
            # Loop over the patches (defined as bars and label) in the bar plot
            for i, p in enumerate(ax.patches):
                # Ignore last patches, because these are labels in plot and not bins
                if i >= len(bins) * nrDatasets:
                    continue
                # get_x gives the x position of the left edge of the patch, and get_width gives the width of the patch
                # This can work as an identifier for the bin
                posPointer = p.get_x() + p.get_width()/2 # Center of bin 
                # Calculate the index_wholePatientUncertaintyMap_binned index based on the x position of the patch. 
                # Create a static list of position for p.get_x() + p.get_width()/2. Have Checked in the plot that these values where correct. 
                posStatic = [-0.2, 0.2, 0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 3.8, 4.2]
                # Compare position of patch with static list and get the index 
                # This will throw error if not found (good for checking that things are the way we expect them to be)
                posIndex = posStatic.index(np.round(posPointer,1))
                # Use this as index to get the bin index (out of the five) in the wholePatientUncertaintyMap_binned
                index_wholePatientUncertaintyMap_binned = math.floor(posIndex / nrDatasets)
                # Get the total number of available voxels for the current bin
                total = wholePatientUncertaintyMap_binned[index_wholePatientUncertaintyMap_binned]
                # Calculate the number of selected voxels for the current bin
                selectedVoxels = p.get_height()
                # Calculate the percentage 
                if total == 0: # Do not divide by zero
                    percentage = 0
                else:
                    percentage = selectedVoxels / total * 100
                # Annotate the patch with the number of selected voxels, the total number of voxels, and the percentage
                # The annotation is placed above the center of the patch
                ax.annotate(f'{int(p.get_height())}\n({int(total)})\n{percentage:.0f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 15), textcoords='offset points', fontsize=8, color='black', weight='bold')

        # Save histogram
        # Folder location is conf.base.histDataOutFolderPath, make sure it exists
        if not os.path.exists(conf.base.histDataOutFolderPath):
            os.makedirs(conf.base.histDataOutFolderPath)
        # Save the plot as a PNG file, with the patient, structure and observer in the filename
        plt.savefig(os.path.join(conf.base.histDataOutFolderPath, f'{patient}_{structure}_histogram_{observer}.png'))
        # Close the plot
        plt.close()

        ##### High uncertainty range analysis #####
        # Create a second image with cutted y axis to focus more on the lower values for uncertainty region 0.4-0.5
        # Change y max value to 5000 and save the plot
        plt.figure(figsize=conf.eval.figSize)
        ymax = 5000 
        plt.ylim(0, ymax)
        # Create the bar plot with black edges
        ax = sns.barplot(x='Bin', y='Count', hue='ObserverStudy', data=df_melted, palette=['blue', 'red'], edgecolor='black')
        # Add labels and title
        plt.xlabel('Uncertainty map bins', fontsize=18)
        plt.ylabel('Voxel frequency count', fontsize=18)
        plt.title(f'{structure} for observer(s) {observer} for patient {patient}', fontsize=20)
        plt.xticks(ticks=np.arange(len(bins)), labels=[f'{b:.2f}-{b + 0.1:.2f}' for b in bins], fontsize=18)
        # Set font size for y ticks
        plt.yticks(fontsize=18)
        plt.legend(loc='upper right', fontsize=18)
        # Save histogram
        # Save the plot as a PNG file, with the patient, structure and observer in the filename
        plt.savefig(os.path.join(conf.base.histDataOutFolderPath, f'{patient}_{structure}_histogram_{observer}_cutted.png'))
        # Close the plot
        plt.close()

        
