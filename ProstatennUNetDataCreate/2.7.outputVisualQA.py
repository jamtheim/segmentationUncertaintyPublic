
# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Code for producing a JPEG visual quality assurance of the data.
# *********************************************************************************

# Import necessary libraries
import os
import numpy as np 
import matplotlib.pyplot as plt
import SimpleITK as sitk
from joblib import Parallel, delayed


def process_patient(basePath, patient): 
    """
    Process a single patient folder and create visual QA images.
    """
    ############### START CONFIGURATION BLOCK ###############
    # Define the base path
    #basePath = '/mnt/mdstore2/Christian/MRIOnlyData/shareData_MRI-only DATERANGE test data/'
    # Define selected patient 
    #patient = 'newAcq_XXX'
    # Define the full subject directory
    subjectDir = os.path.join(basePath, patient)


    # Define names of the MRI and sCT sub directories used in the data 
    MRIDir = 'MR_StorT2'
    sCTDir = 'sCT'
    nnUNetOutputDir = 'nnUNetOutput'
    observerDataDir = 'observerData'
    # Define the paths to the MRI, sCT volumes and dose volumes 
    MRIVolumePath = os.path.join(subjectDir, MRIDir, 'image.nii.gz')
    sCTVolumePath = os.path.join(subjectDir, sCTDir, 'image.nii.gz')
    sCTReg2MRIVolumePath = os.path.join(subjectDir, sCTDir, 'image_reg2MRI.nii.gz')
    sCTDosePath = os.path.join(subjectDir, sCTDir, 'dose_original.nii.gz')
    sCTDoseInterpolatedPath = os.path.join(subjectDir, sCTDir, 'dose_interpolated.nii.gz')
    MRIDoseInterpolatedPath = os.path.join(subjectDir, MRIDir, 'dose_interpolated.nii.gz')
    # Define path to fiducial point text file and segmentation mask
    fiducialTxtPath = os.path.join(subjectDir, MRIDir, 'MRI_T2_DICOM_coords_fiducials.txt')
    fiducialMaskPath = os.path.join(subjectDir, MRIDir, 'mask_MRI_T2_coords_fiducials.nii.gz')
    # Define paths for some sample MRI structures
    MRICTVPath = os.path.join(subjectDir, MRIDir, 'mask_CTVT_427.nii.gz')
    MRIRectumPath = os.path.join(subjectDir, MRIDir, 'mask_Rectum.nii.gz')
    MRIFemoralHeadLPath = os.path.join(subjectDir, MRIDir, 'mask_FemoralHead_L.nii.gz')
    MRIFemoralHeadRPath = os.path.join(subjectDir, MRIDir, 'mask_FemoralHead_R.nii.gz')
    # Define paths for some sample sCT structures
    sCTCTVPath = os.path.join(subjectDir, sCTDir, 'mask_CTVT_427.nii.gz')
    sCTRectumPath = os.path.join(subjectDir, sCTDir, 'mask_Rectum.nii.gz')
    sCTFemoralHeadLPath = os.path.join(subjectDir, sCTDir, 'mask_FemoralHead_L.nii.gz')
    sCTFemoralHeadRPath = os.path.join(subjectDir, sCTDir, 'mask_FemoralHead_R.nii.gz')
    # Define paths for two different observer segmentations (total four available, obsB-E)
    MRICTVObsCPath = os.path.join(subjectDir, MRIDir, observerDataDir, 'mask_CTVT_427_step1_obsC.nii.gz') 
    MRICTVObsDPath = os.path.join(subjectDir, MRIDir, observerDataDir, 'mask_CTVT_427_step1_obsD.nii.gz')

    # Define folder for output png files
    scriptFolder = os.path.dirname(os.path.realpath(__file__))
    pngOutputFolder = os.path.join(scriptFolder, 'pngQAoutput')
    # Create the output folder if it does not exist
    os.makedirs(pngOutputFolder, exist_ok=True)
    ############### END CONFIGURATION BLOCK ###############

    # Load the volumes using SimpleITK
    # Image volumes
    MRIImage = sitk.ReadImage(MRIVolumePath)
    sCTImage = sitk.ReadImage(sCTVolumePath)
    sCTReg2MRIImage = sitk.ReadImage(sCTReg2MRIVolumePath)
    sCTDoseImage = sitk.ReadImage(sCTDosePath)
    sCTDoseInterpolatedImage = sitk.ReadImage(sCTDoseInterpolatedPath)
    MRIDoseInterpolatedImage = sitk.ReadImage(MRIDoseInterpolatedPath)
    # Fiducial points
    fiducialImage  = sitk.ReadImage(fiducialMaskPath)
    # Structure volumes MRI
    MRICTVImage = sitk.ReadImage(MRICTVPath)
    MRIRectumImage = sitk.ReadImage(MRIRectumPath)
    MRIFemoralHeadLImage = sitk.ReadImage(MRIFemoralHeadLPath)
    MRIFemoralHeadRImage = sitk.ReadImage(MRIFemoralHeadRPath)
    # Structure volumes sCT 
    sCTCTVImage = sitk.ReadImage(sCTCTVPath)
    sCTRectumImage = sitk.ReadImage(sCTRectumPath)
    sCTFemoralHeadLImage = sitk.ReadImage(sCTFemoralHeadLPath)
    sCTFemoralHeadRImage = sitk.ReadImage(sCTFemoralHeadRPath)


    # Get the numpy arrays of the images from the SimpleITK objects
    # Image volumes
    MRIArray = sitk.GetArrayFromImage(MRIImage)
    sCTArray = sitk.GetArrayFromImage(sCTImage)
    sCTReg2MRIArray = sitk.GetArrayFromImage(sCTReg2MRIImage)
    sCTDoseArray = sitk.GetArrayFromImage(sCTDoseImage)
    sCTDoseInterpolatedArray = sitk.GetArrayFromImage(sCTDoseInterpolatedImage)
    MRIDoseInterpolatedArray = sitk.GetArrayFromImage(MRIDoseInterpolatedImage)
    # Fiducial points
    fiducialArray = sitk.GetArrayFromImage(fiducialImage)
    # Structure volumes MRI
    MRICTVArray = sitk.GetArrayFromImage(MRICTVImage)
    MRIRectumArray = sitk.GetArrayFromImage(MRIRectumImage)
    MRIFemoralHeadLArray = sitk.GetArrayFromImage(MRIFemoralHeadLImage)
    MRIFemoralHeadRArray = sitk.GetArrayFromImage(MRIFemoralHeadRImage)
    # Structure volumes sCT
    sCTCTVArray = sitk.GetArrayFromImage(sCTCTVImage)
    sCTRectumArray = sitk.GetArrayFromImage(sCTRectumImage)
    sCTFemoralHeadLArray = sitk.GetArrayFromImage(sCTFemoralHeadLImage)
    sCTFemoralHeadRArray = sitk.GetArrayFromImage(sCTFemoralHeadRImage)

    # Get the middle slice of the MRI and sCT volumes each (slices first in the numpy array)
    middleSliceMRI = MRIArray.shape[0] // 2
    middleSliceSCT = sCTArray.shape[0] // 2
    middleSliceDose = sCTDoseArray.shape[0] // 2
    # Print the middle slice numbers
    print(f'Middle slice MRI: {middleSliceMRI}')
    print(f'Middle slice sCT: {middleSliceSCT}')
    print(f'Middle slice sCT dose: {middleSliceDose}')

    # Plot MRI with structures and dose as blended overlay
    plt.imshow(MRIArray[middleSliceMRI, :, :], cmap='gray')
    plt.imshow(MRIDoseInterpolatedArray[middleSliceMRI, :, :], cmap='rainbow', alpha=0.6)
    plt.imshow(MRICTVArray[middleSliceMRI, :, :], cmap='jet', alpha=0.2)
    plt.imshow(MRIRectumArray[middleSliceMRI, :, :], cmap='jet', alpha=0.2)
    plt.imshow(MRIFemoralHeadLArray[middleSliceMRI, :, :], cmap='jet', alpha=0.2)
    plt.imshow(MRIFemoralHeadRArray[middleSliceMRI, :, :], cmap='jet', alpha=0.2)
    plt.title('MRI')
    plt.axis('off')
    plt.savefig(os.path.join(pngOutputFolder, patient + '_MRI.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.clf()

    # Plot sCT registered to MRI with structures as blended overlay
    # Notice use of MRI structures for overlay
    plt.imshow(sCTReg2MRIArray[middleSliceMRI, :, :], cmap='gray')
    plt.imshow(MRICTVArray[middleSliceMRI, :, :], cmap='jet', alpha=0.2)
    plt.imshow(MRIRectumArray[middleSliceMRI, :, :], cmap='jet', alpha=0.2)
    plt.imshow(MRIFemoralHeadLArray[middleSliceMRI, :, :], cmap='jet', alpha=0.2)
    plt.imshow(MRIFemoralHeadRArray[middleSliceMRI, :, :], cmap='jet', alpha=0.2)
    plt.title('sCT registered to MRI with example structures')
    plt.axis('off')
    plt.savefig(os.path.join(pngOutputFolder, patient + '_sCTReg2MRI.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.clf()

    # Plot original sCT with example structures and dose as blended overlay
    plt.imshow(sCTArray[middleSliceSCT, :, :], cmap='gray')
    plt.imshow(sCTDoseInterpolatedArray[middleSliceSCT, :, :], cmap='rainbow', alpha=0.5)
    plt.imshow(sCTCTVArray[middleSliceSCT, :, :], cmap='jet', alpha=0.2)
    plt.imshow(sCTRectumArray[middleSliceSCT, :, :], cmap='jet', alpha=0.2)
    plt.imshow(sCTFemoralHeadLArray[middleSliceSCT, :, :], cmap='jet', alpha=0.2)
    plt.imshow(sCTFemoralHeadRArray[middleSliceSCT, :, :], cmap='jet', alpha=0.2)
    plt.title('sCT')
    plt.axis('off')
    plt.savefig(os.path.join(pngOutputFolder, patient + '_sCT.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.clf()

    # Plot registration between MRI and sCT
    plt.imshow(sCTReg2MRIArray[middleSliceSCT, :, :], cmap='gray')
    plt.imshow(MRIArray[middleSliceMRI, :, :], cmap='jet', alpha=0.5)
    plt.title('sCT registered to MRI with MRI overlay blended')
    plt.axis('off')
    plt.savefig(os.path.join(pngOutputFolder, patient + '_registration.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.clf()


def main(folder_location):
    """
    Main function to process all patient folders.

    Args:
        folder_location (str): Path to the root folder containing patient data.
    """
    print(f"Scanning folder: {folder_location}")

    # Process all patient folders in parallel
    Parallel(n_jobs=30, verbose=10)(delayed(process_patient)(folder_location, patient_folder) for patNr, patient_folder in enumerate(sorted(os.listdir(folder_location))))


# Example usage
if __name__ == "__main__":

    # Set data type
    dataType = 'Test'

    # Set path to the folder containing the data
    if dataType == 'Train':
        folder_location = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/shareData_MRI-only from DATE')
    if dataType == 'Test':
        folder_location = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/shareData_MRI-only DATERANGE test data')
    
    # Call the main function
    main(folder_location)
