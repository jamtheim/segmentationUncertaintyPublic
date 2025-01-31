# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: This script contains code for calculating inter-observer 
# differences between step 1 and step 2 for both CTV and Rectum.
# This is done in aggregated way by analysing reported and saved geometrical metrics.
# As we have non-normal distribution of data we use 
# Wilcoxon Signed-Rank Test and Fligner-Killeen Test.
# *********************************************************************************

import os
import csv
import numpy as np
import pandas as pd
import scipy.stats as stats
from commonConfig import commonConfigClass
from evalDataMethods import evalDataMethodsClass
import warnings

# Suppress specific warning (easier to read output)
warnings.filterwarnings("ignore", message="Exact p-value calculation does not work if there are zeros. Switching to normal approximation.")

# Load configuration
conf = commonConfigClass() # Configuration for the project
# Load methods
evalData = evalDataMethodsClass() # Functions for converting DICOM to Nifti data


class evalDataMethodsClass:

    def loadCSVData(self, observer, patient, structure, step, description, metric):
        """
        Loads CSV data for a given structure, step, observer, and patient for a specific metric.

        Args:
            observer (str): The observer identifier (e.g., obsB, obsC, obsD, obsE)
            patient (str): The patient identifier (e.g., pat1, pat2, etc.)
            structure (str): The anatomical structure (e.g., CTV, Rectum)
            step (int): The step number (1 or 2)
            metric (str): The metric to extract (e.g., HD, DSC)
            description (str): A description of the comparison for the file name.

        Returns:
            float: The value of the metric for the given observer, patient, and step.
        """
        # Define the file path based on the function inputs
        filePath = os.path.join(conf.base.evalDataEditedOutFolderPath, 
                                f"{patient}_{observer}_{structure}_step{step}_{description}.csv")
        # Init data dictionary
        dataDict = {}
        # Open the file
        with open(filePath, 'r', newline='') as f:
            reader = csv.reader(f)
            # Loop through each row in the CSV file
            for row in reader:
                # Assume the first column is the key and the second column is the value
                if len(row) == 2:
                    key, value = row
                    # APL has an array, ignore it as it is not a float value
                    if key != 'APL': 
                        dataDict[key] = float(value)
        # Return the value for the given metric
        return dataDict[metric]
      

    def loadAllData(self, structure, step, description, metric):
        """
        Loads data for a given structure, step, and metric across all observers and patients.

        Args:
            structure (str): The anatomical structure (e.g., CTV, Rectum)
            step (int): The step number (1 or 2)
            description (str): A description of the comparison for the file name.
            metric (str): The metric to extract (e.g., HD, DSC)

        Returns:
            pd.DataFrame: A DataFrame containing data for each patient and observer.
        """
        # Initialize lists for storing data
        data = []
    
        # Loop through all patients and observers
        for patient in conf.eval.patients:
            for observer in conf.eval.observers:
                value = self.loadCSVData(observer, patient, structure, step, description, metric)
                data.append([patient, observer, step, description, value])
        
        # Create a DataFrame from the collected data
        return pd.DataFrame(data, columns=['Patient', 'Observer', 'Step', 'Description', metric])


    def performStatisticalTest(self, df, metric, results_df, description):
        """
        Perform Wilcoxon Signed-Rank Test to test if two related paired samples come from the same distribution, i.e from Step1 and Step2 for each observer. 
        Fligner-Killeen test measures if the spread (inter-observer) among the four observers differs between Step1 and Step2.

        Args:
            df (pd.DataFrame): DataFrame containing observer data for patients and steps.
            metric (str): The metric to test (e.g., HD, DSC).
            results_df (pd.DataFrame): DataFrame to store test results.

        Returns:
            None: Prints the test results.
        """
        # Pivot data for each observer, separating steps
        data_pivot = df.pivot_table(index='Patient', columns=['Observer', 'Step'], values=metric)

        # Perform Wilcoxon test for each observer (compare Step 1 vs Step 2)
        for observer in conf.eval.observers:
            step1_data = [] # Empty list for Step 1 data
            step2_data = [] # Empty list for Step 2 data
            step1_data = data_pivot[observer][1].dropna()  # Data for Step 1
            step2_data = data_pivot[observer][2].dropna()  # Data for Step 2
            # Assert that it is 35 values for each observer in each step
            assert len(step1_data) == 35, f"Observer {observer} has {len(step1_data)} values for Step 1."
            assert len(step2_data) == 35, f"Observer {observer} has {len(step2_data)} values for Step 2."
            # Wilcoxon Signed-Rank Test per observer
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(step1_data, step2_data, alternative="two-sided", method='auto', zero_method="wilcox", correction=False)
            # Calculate median of the difference between Step 1 and Step 2
            median_diff = np.median(step1_data-step2_data)
            # Calculate the std of the difference
            std_diff = np.std(step1_data-step2_data)
            # Calculate min and max of the difference
            min_diff = np.min(step1_data-step2_data)
            max_diff = np.max(step1_data-step2_data)
            # Store the results in the table (median difference plus/minus std [min, max] and (p-value)
            results_df.at[observer, metric] = f"{median_diff:.2f}+/-{std_diff:.2f} [{min_diff:.2f}, {max_diff:.2f}] (p={wilcoxon_p:.2f})" 
            #print(f"Wilcoxon Signed-Rank Test for observer {observer}: statistic={wilcoxon_stat}, p-value={wilcoxon_p}")
            #print(f"Median difference: {median_diff}")
            #if wilcoxon_p < 0.05:
            #    print(f"Statistically significant difference detected between Step 1 and Step 2 for observer {observer} (Wilcoxon).")
            #    print(' ')
            #else:
            #   print(f"No statistically significant difference detected between Step 1 and Step 2 for observer {observer} (Wilcoxon).")
            #   print(' ')
            # Get statistics for step 1 only
            step1_median = np.median(step1_data)
            step1_std = np.std(step1_data)
            step_1_min = np.min(step1_data)
            step_1_max = np.max(step1_data)
            # Print description
            print(f"Description: {description}")
            # Print
            print(f"Step 1 median - observer {observer}: {step1_median:.2f}+/-{step1_std:.2f} [{step_1_min:.2f}, {step_1_max:.2f}]")
            # Include description in the results table

            # For step 2
            step2_median = np.median(step2_data)
            step2_std = np.std(step2_data)
            step_2_min = np.min(step2_data)
            step_2_max = np.max(step2_data)
            # Print
            print(f"Step 2 median - observer {observer}: {step2_median:.2f}+/-{step2_std:.2f} [{step_2_min:.2f}, {step_2_max:.2f}]")

        
        ### Perform Wilcoxon over the whole group of observers ### 
        ### Also perform the Fligner-Killeen test for homogeneity of variances across observers ###
        
        # Flatten data for Step 1 and Step 2 across all observers
        step1_data = [] # Empty list for Step 1 data
        step2_data = [] # Empty list for Step 2 data
        step1_data = [data_pivot[observer][1].dropna().values for observer in conf.eval.observers]
        step2_data = [data_pivot[observer][2].dropna().values for observer in conf.eval.observers]
        # Assess correct number of values for each observer 
        for i, (step1_values, step2_values) in enumerate(zip(step1_data, step2_data)):
            assert len(step1_values) == 35, f"Observer {conf.eval.observers[i]} has {len(step1_values)} values for Step 1."
            assert len(step2_values) == 35, f"Observer {conf.eval.observers[i]} has {len(step2_values)} values for Step 2."
        # Make a list of all values for each step
        step1_data = [item for sublist in step1_data for item in sublist]
        step2_data = [item for sublist in step2_data for item in sublist]
        # Assert number of values for each step
        assert len(step1_data) == 140, f"Step 1 has {len(step1_data)} values."
        assert len(step2_data) == 140, f"Step 2 has {len(step2_data)} values."
        # Create array
        step1_data = np.array(step1_data)
        step2_data = np.array(step2_data)

        # Calculate median difference between Step 1 and Step 2 for all observers combined
        median_diff = np.median(step1_data-step2_data)
        std = np.std(step1_data-step2_data)
        min_diff = np.min(step1_data-step2_data)
        max_diff = np.max(step1_data-step2_data)

        # Perform Wilcoxon Signed-Rank Test for all observers combined
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(step1_data, step2_data, alternative="two-sided", method='auto', zero_method="wilcox", correction=False)
        # Store Wilcoxon test result in "allObservers" row
        results_df.at['allObs', metric] = f"{median_diff:.2f}+/-{std:.2f} [{min_diff:.2f}, {max_diff:.2f}] (p={wilcoxon_p:.2f})"
        # Print
        print(f"Wilcoxon Signed-Rank Test for all observers combined: statistic={wilcoxon_stat}, p-value={wilcoxon_p}")
        
        # Perform Fligner-Killeen test for equality of variance of step1 and step2 data 
        # Fligner's test tests the null hypothesis that all input samples
        # are from populations with equal variances. 
        fligner_stat, fligner_p = stats.fligner(step1_data, step2_data, center='median')
        print(f"Fligner-Killeen test result: statistic={fligner_stat}, p-value={fligner_p}")
        
        # Variance for each step and difference
        varStep1 = np.var(step1_data)
        varStep2 = np.var(step2_data)
        var_Diff = varStep1 - varStep2

        print(' ')
        # Store Fligner-Killeen test result in "interObs" row
        results_df.at['interObs', metric] = f"p={fligner_p:.2f}, varDiff={var_Diff:.2f})"

        #print(f"Median difference: {median_diff}")
        #if fligner_p < 0.05:
        #    print(f"Significant difference in variance detected between Step 1 and Step 2 across observers (Fligner-Killeen).")
        #    print(' ')
        #else:
        #    print(f"No significant difference in variance detected between Step 1 and Step 2 across observers (Fligner-Killeen).")
        #    print(' ')  


    def analyzeMetric(self, structure, description, metric, results_df):
        """
        Main function to load data and perform statistical analysis for a given metric.

        Args:
            structure (str): The anatomical structure (e.g., CTV, Rectum)
            metric (str): The metric to analyze (e.g., HD, DSC)
            results_df (pd.DataFrame): DataFrame to store test results.

        Returns:
            None: Outputs the analysis results.
        """
        # Print the structure and metric being analyzed
        print(f"Analyzing the {metric} metric for the {structure} structure.")
        # Load data for step 1 and step 2
        df_step1 = self.loadAllData(structure, step=1, description=description, metric=metric)
        df_step2 = self.loadAllData(structure, step=2, description=description, metric=metric)
        # Combine data for both steps
        df_combined = pd.concat([df_step1, df_step2])
        # Perform the statistical test
        self.performStatisticalTest(df_combined, metric, results_df, description)


# This is the main part of the script
# Usage
eval_data = evalDataMethodsClass()
# Define array of structures and metrics to analyze
structures = ["CTV", 'Rectum']
metrics = ["DSC", "SurfaceDSC", "HD", "HD95", "VolumeDifference", "ASD_ref2obs", "TotalAPL"]

# Analyzing the metric for each structure and metric
for currStructure in structures:
    print('')
    print('Analyzing structure: ' + currStructure)
    # Create a results DataFrame to store the test results
    results_df =[] 
    results_df = pd.DataFrame(index=conf.eval.observers + ['interObs'], columns=metrics)
    results_df = pd.DataFrame(index=conf.eval.observers + ['allObs'], columns=metrics)
   
    for currMetric in metrics:
        eval_data.analyzeMetric(structure=currStructure, description='segMetricsVsRef', metric=currMetric, results_df=results_df)
     
    # Display the final results table
    print("\nFinal results table for structure " + currStructure)
    print(' ' ) 
    # print the results table. Do not cut the printout
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(results_df)
    # Save the results table to a CSV file, name after structure
    results_df.to_csv(os.path.join(conf.base.evalDataEditedOutFolderPath, f"{currStructure}_interObsDiffAgg.csv"))

    # Metric data can alse be acquired in the plots from
    #/mnt/mdstore2/Christian/MRIOnlyData/inference/observerDataEditedAnalysis/plotData/
