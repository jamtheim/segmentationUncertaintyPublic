# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Configuration file for the observer data evaluation pipeline. 
# Controls the evaluation of observer data for the project.
# *********************************************************************************

import os
import numpy as np

###### Check override options below to change when all data is collected ######
# Search for #####

class commonConfigClass():
    """
    Class describing the common configuration used in the project.
    """

    def __init__ (self):
        """
        Initialize the common configuration class.
        """
        self.base, self.eval, self.debug = self.initializeConfiguration()


    class base:
        """
        Empty class to define base related configuration.
        """
        pass


    class eval:
        """
        Empty class to define evaluation related configuration.
        """
        pass

    class debug:
        """
        Empty class to define debug related configuration.
        """
        pass


    def initializeConfiguration(self):
        """
        Initialize the configurations 

        Returns:
            tuple: A tuple containing the base, and eval configurations.
        """

        # Init configuration blocks 
        base = self.base()
        eval = self.eval()
        debug = self.debug()
    
        # Set base configuration
        base.infDataBaseFolderPath = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/inference/')
        base.infObsFolderName = 'MRI-only_test_data'
        base.infEclipseDataFolderName = 'exportEclipse'
        base.infCTVFolderName = 'inference_NiftiSeg_out_TaskNumber_110' # Contains original segmentation, uncertainty and softmax 
        base.infRectumFolderName = 'inference_NiftiSeg_out_TaskNumber_111' # Contains original segmentation, uncertainty and softmax 
        base.infImageSeriesName = 'MR_StorT2'
        base.infStep1FolderName = 'Step1'
        base.infStep2FolderName = 'Step2'

        # Observer data base folder 
        base.obsDataEditedBaseFolder = 'observerDataEdited'
        base.obsDataEditedFolderPath = os.path.join(base.infDataBaseFolderPath, base.obsDataEditedBaseFolder)
        # This folder will contain 
        # ./obsN/patientX/step1Edited/Rectum1/ ./obsN/patientX/step1Edited/CTV1/
        # ./obsN/patientX/step2Edited/Rectum2/ ./obsN/patientX/step2Edited/CTV2/

        # Path can be created from the variables below
        base.obsFolderName = 'obs'
        base.niftiFolderName = 'Nifti'
        base.step1FolderName = 'step1Edited'
        base.step2FolderName = 'step2Edited'
        base.referenceFolderName = 'refEclipse'
        base.step1CTVFolderName = 'CTV1'
        base.step1RectumFolderName = 'Rectum1'
        base.step2CTVFolderName = 'CTV2'
        base.step2RectumFolderName = 'Rectum2'

        # Set output folder for evaluation data in npz format
        base.evalDataEditedOutBaseFolder = 'observerDataEditedAnalysis'
        base.evalDataEditedOutFolderPath = os.path.join(base.infDataBaseFolderPath, base.evalDataEditedOutBaseFolder)
        # Set output folder for histogram data 
        base.histDataOutBaseFolder = 'histogramData'
        base.histDataOutFolderPath = os.path.join(base.evalDataEditedOutFolderPath, base.histDataOutBaseFolder)
        # Set output folder for plot data
        base.plotDataOutBaseFolder = 'plotData'
        base.plotDataOutFolderPath = os.path.join(base.evalDataEditedOutFolderPath, base.plotDataOutBaseFolder)

        # Set evaluation options 
        # Create list with pat1 to pat35 using loop
        eval.patients = ['test_unc_pat' + str(i) for i in range(1, 36)]

        # Create list with observers to evaluate
        eval.observers = ['obsB', 'obsC', 'obsD', 'obsE']
        # Create list for structures
        eval.structures = ['CTV', 'Rectum']
        # Create list for steps 
        eval.steps = [1,2]
        

        # Set threshold for STAPLE
        eval.STAPLEThreshold = 0.5
        # Set number of geometric metrics to calculate, needed for QA of results
        eval.nrGeometricMetrics = 17

        # Thresholds for contour evaluation
        eval.apl_distance_threshold = 1
        eval.surface_dice_threshold = 1
        eval.surface_overlap_threshold = 1
        # Assertions for data input
        eval.numberTotalSlices = 88
        eval.sliceThickness = 2.5 # mm

        # Set number of CPU to use in multiprocessing
        eval.nrCPU = 35
        # Set bin edges for histogram
        eval.bin_edges = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        # Set fig size
        eval.figSize = (11, 7)

        # Plot settings
        eval.markersMetricAllObservers = {'obsB': 'o', 'obsB': 'o', 'obsC': 'o', 'obsD': 'o', 'obsE': 'o'}
        eval.colorsMetricAllObservers = {'obsA': 'yellow', 'obsB': 'red', 'obsC': 'blue', 'obsD': 'green', 'obsE': 'black'}
        eval.plotSizeMetricAllObservers = 50 

        # Statistics
        eval.keysWilcoxonTest = ['VolumeDifference', 'VolumeRatio'] # Keys to perform Wilcoxon test on
        eval.referencePvalue = 0.05

        # Set flag for QA
        eval.QA = True
        # Set manufacturer name for RT structs
        eval.obsDataManufacturer = 'Varian Medical Systems'
        eval.refDataManufacturer = 'Qurit' 

        # Set units for the metrics and info on how to validate the metrics
        eval.unitDict = {
            "VolumeRef": 'cm3', # Calc validated towards Eclipse
            "VolumeObs": 'cm3', # Calc validated towards Eclipse
            "VolumeRatio": '',  # No calc validation neded
            "VolumeDifference": 'cm3', # Calc validated towards Eclipse
            "MeanAPL": 'mm', # Seems reasonable when counting by hand
            "TotalAPL": 'mm', # Seems reasonable when counting by hand
            "Sensitivity": '',
            "Specificity": '',
            "DSC": '', # Calc validated in deep mind package
            "HD": 'mm', # Calc validated in deep mind package
            "HD95": 'mm', # Calc validated in deep mind package
            "HD99": 'mm', # Calc validated in deep mind package
            "SurfaceDSC": '', # Calc validated in deep mind package
            "SurfaceOverlap_ref2obs": '', # Calc validated in deep mind package
            "SurfaceOverlap_obs2ref": '', # Calc validated in deep mind package
            "ASD_ref2obs": 'mm', # Calc validated in deep mind package
            "ASD_obs2ref": 'mm' # Calc validated in deep mind package
            }

        # Patient based eval configuration and expected values
        # These patients are excepted in the QA RT struct manufacturer check because no changes were needed to the structures
        # Verified with Google Sheet observer grading. 
        # This provides a list of structures that were unchanged with respect to reference. 
        # It also shows and overview of differences between step 1 and step 2.
        eval.ignoreManufactQAStep1CTV = [
            'test_unc_pat7_obsC',
            'test_unc_pat25_obsC',
            'test_unc_pat16_obsD',
            'test_unc_pat10_obsE',
            'test_unc_pat33_obsE',
         ] 
        eval.ignoreManufactQAStep1Rectum = [
            'test_unc_pat16_obsD',
            'test_unc_pat3_obsE',
            'test_unc_pat4_obsE',
            'test_unc_pat7_obsE',
            'test_unc_pat16_obsE',
            'test_unc_pat17_obsE',
            'test_unc_pat23_obsE',
            'test_unc_pat24_obsE',
            'test_unc_pat25_obsE',
            'test_unc_pat29_obsE',
         ] 
        
        eval.ignoreManufactQAStep2CTV = [
            'test_unc_pat7_obsC', 
            'test_unc_pat22_obsC', 
            'test_unc_pat6_obsE',
            'test_unc_pat7_obsE',
            'test_unc_pat8_obsE',
            'test_unc_pat20_obsE', 
            'test_unc_pat24_obsE', 
            'test_unc_pat27_obsE', 
            'test_unc_pat33_obsE', 
            'test_unc_pat35_obsE',
        ]

        eval.ignoreManufactQAStep2Rectum = [
            'test_unc_pat15_obsC', 
            'test_unc_pat4_obsE',
            'test_unc_pat6_obsE',
            'test_unc_pat7_obsE',
            'test_unc_pat17_obsE',
            'test_unc_pat24_obsE', 
            'test_unc_pat25_obsE', 
            'test_unc_pat26_obsE', 
        ]

        # Create a dictionary to set maximum y-value for histogram plots
        # This is defined per patient 
        eval.plotHistogramFromMaskMaxYvalue = {
            'test_unc_showcase7': 3000,
            'test_unc_pat1': 6000,
            'test_unc_pat2': 6000,
            'test_unc_pat3': 6000,
            'test_unc_pat4': 6000,
            'test_unc_pat5': 6000,
            'test_unc_pat6': 6000,
            'test_unc_pat7': 6000,
            'test_unc_pat8': 6000,
            'test_unc_pat9': 6000,
            'test_unc_pat10': 6000,
            'test_unc_pat11': 10000,
            'test_unc_pat12': 10000,
            'test_unc_pat13': 6000,
            'test_unc_pat14': 6000,
            'test_unc_pat15': 6000,
            'test_unc_pat16': 10000,
            'test_unc_pat17': 6000,
            'test_unc_pat18': 10000,
            'test_unc_pat19': 6000,
            'test_unc_pat20': 6000,
            'test_unc_pat21': 6000,
            'test_unc_pat22': 6000,
            'test_unc_pat23': 6000,
            'test_unc_pat24': 6000,
            'test_unc_pat25': 6000,
            'test_unc_pat26': 14000,
            'test_unc_pat27': 10000,
            'test_unc_pat28': 6000,
            'test_unc_pat29': 6000,
            'test_unc_pat30': 6000,
            'test_unc_pat31': 6000,
            'test_unc_pat32': 10000,
            'test_unc_pat33': 10000,
            'test_unc_pat34': 6000,
            'test_unc_pat35': 6000,
            'All': 45000
        }
        # Create a dictionary to set maximum y-value for scatter plots
        # This is defined per patient 
        eval.plotVoxelsAboveThresholdsMaxYvalue = {
            'test_unc_showcase7': 5000,
            'test_unc_pat1': 8000,
            'test_unc_pat2': 10000,
            'test_unc_pat3': 8000,
            'test_unc_pat4': 6000,
            'test_unc_pat5': 10000,
            'test_unc_pat6': 6000,
            'test_unc_pat7': 6000,
            'test_unc_pat8': 10000,
            'test_unc_pat9': 8000,
            'test_unc_pat10': 6000,
            'test_unc_pat11': 10000,
            'test_unc_pat12': 10000,
            'test_unc_pat13': 6000,
            'test_unc_pat14': 11000,
            'test_unc_pat15': 8000,
            'test_unc_pat16': 14000,
            'test_unc_pat17': 8000,
            'test_unc_pat18': 14000,
            'test_unc_pat19': 6000,
            'test_unc_pat20': 6000,
            'test_unc_pat21': 6000,
            'test_unc_pat22': 8000,
            'test_unc_pat23': 6000,
            'test_unc_pat24': 8000,
            'test_unc_pat25': 6000,
            'test_unc_pat26': 20000,
            'test_unc_pat27': 6000,
            'test_unc_pat28': 6000,
            'test_unc_pat29': 6000,
            'test_unc_pat30': 6000,
            'test_unc_pat31': 6000,
            'test_unc_pat32': 12000,
            'test_unc_pat33': 30000,
            'test_unc_pat34': 6000,
            'test_unc_pat35': 8000,
            'All': 5000*35
        }


        #### TEST DEV OPTIONS ### 
        # Set debug options 
        debug.fakeAllSegData = False # Set to True to use fake segmentation data for testing

        # For End to End Testing enable this, else keep False
        endToendTestMode = False

        if endToendTestMode == True:    
            base.infObsFolderName = 'MRI-only_showcase'
            eval.QA = False # For it not to stop at QA
            base.obsDataEditedBaseFolder = 'observerDataEndToEndTest'
            base.obsDataEditedFolderPath = os.path.join(base.infDataBaseFolderPath, base.obsDataEditedBaseFolder)
            eval.patients = ['test_unc_showcase7']
            eval.observers = ['obsB']
            # Set output folder for evaluation data in npz format
            base.evalDataEditedOutBaseFolder = 'observerDataEndToEndTestAnalysis'
            base.evalDataEditedOutFolderPath = os.path.join(base.infDataBaseFolderPath, base.evalDataEditedOutBaseFolder)
            # Set output folder for histogram data 
            base.histDataOutBaseFolder = 'histogramData'
            base.histDataOutFolderPath = os.path.join(base.evalDataEditedOutFolderPath, base.histDataOutBaseFolder)
            # Set output folder for plot data
            base.plotDataOutBaseFolder = 'plotData'
            base.plotDataOutFolderPath = os.path.join(base.evalDataEditedOutFolderPath, base.plotDataOutBaseFolder)

        ### END SECTION FOR TESTING ###


        # Return the configuration
        return base, eval, debug
        
