# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physicist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Collect volumes from structures in Nifti files and plot a histogram.
# Also perform a Shapiro-Wilk test for normality. Histograms are saved in 'volumeHistogram'.
# *********************************************************************************

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import shapiro

def validateBinaryMask(niftiData):
    """
    Validate that the input Nifti data is a binary mask.

    Args:
        niftiData (numpy.ndarray): 3D array containing Nifti image data.

    Raises:
        ValueError: If the data is not binary (contains values other than 0 and 1).
    """
    unique_values = np.unique(niftiData)
    if not np.array_equal(unique_values, [0, 1]):
        raise ValueError("The Nifti file does not contain a binary mask.")

def getVoxelVolume(header):
    """
    Calculate the voxel volume from the Nifti header.

    Args:
        header (nib.Nifti1Header): Nifti file header object.

    Returns:
        float: Volume of a single voxel in mm^3.
    """
    voxel_dims = header.get_zooms()
    # Multiply the dimensions to get the volume
    return np.prod(voxel_dims)

def calculateMaskVolume(filePath):
    """
    Calculate the volume of a binary mask from a Nifti file.

    Args:
        filePath (str): Path to the Nifti file.

    Returns:
        float: Volume of the binary mask in mm^3.
    """
    nifti_image = nib.load(filePath)
    data = nifti_image.get_fdata()
    # validateBinaryMask(data)
    voxel_volume = getVoxelVolume(nifti_image.header)
    mask_volume = np.sum(data) * voxel_volume
    return mask_volume

def ensureOutputDirectoryExists():
    """
    Ensure the output directory 'volumeHistogram' exists.

    Returns:
        str: Path to the output directory.
    """
    output_dir = os.path.join('/mnt/mdstore2/Christian/YngreAlf/SegUnc/src/ProstatennUNetDataCreate', "volumeHistogram")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def collectVolumes(folderLocation, fileName, subFolder):
    """
    Recursively search for files matching the given name in a folder, calculate volumes, and collect them in a list.

    Args:
        folderLocation (str): Root folder to search for files.
        fileName (str): Name of the files to search for.
        subFolder (str): Subfolder that must be part of the file path.

    Returns:
        list: List of volumes (in mm^3) for each matching file.
    """
    volumes = []
    matching_files = []
    for root, _, files in os.walk(folderLocation):
        if subFolder in root:  # Ensure the subfolder exists in the current path
            for file in files:
                if file == fileName:
                    filePath = os.path.join(root, file)
                    matching_files.append(filePath)

    assert len(matching_files) > 0, f"No files matching '{fileName}' in subfolder '{subFolder}' were found."
    
    total_files = len(matching_files)
    for idx, filePath in enumerate(matching_files, start=1):
        print(f"Processing {filePath} ({idx}/{total_files})...")
        volume = calculateMaskVolume(filePath)
        volumes.append(volume)
    
    return volumes

def plotHistogramAndTest(volumes, fileName):
    """
    Create a histogram of the volumes, perform a Shapiro-Wilk test for normality, and save the histogram.

    Args:
        volumes (list): List of volumes to analyze.
        fileName (str): Name of the structure analyzed (for labeling purposes).

    Returns:
        None
    """
    output_dir = ensureOutputDirectoryExists()
    hist_file_path = os.path.join(output_dir, f"hist_{fileName.replace('.nii.gz', '')}.png")
    
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(volumes, bins=15, alpha=0.75, color='blue', edgecolor='black')
    plt.title(f"Histogram of Volumes for {fileName}")
    plt.xlabel("Volume (mm^3)")
    plt.ylabel("Frequency")
    plt.savefig(hist_file_path)
    plt.close()
    print(f"Histogram saved to: {hist_file_path}")

    # Perform Shapiro-Wilk test
    stat, p_value = shapiro(volumes)
    print(f"Shapiro-Wilk Test for {fileName}: Statistic={stat:.4f}, p-value={p_value:.4f}")
    if p_value > 0.05:
        print("The volumes appear to be normally distributed.")
    else:
        print("The volumes do not appear to be normally distributed.")

def main(folderLocation, fileName, subFolder):
    """
    Main function to calculate and analyze mask volumes.

    Args:
        folderLocation (str): Root folder to search for files.
        fileName (str): Name of the files to search for.
        subFolder (str): Subfolder that must exist in the search path.

    Returns:
        None
    """
    print(f"Searching for files named '{fileName}' in '{folderLocation}' within subfolder '{subFolder}'...")
    volumes = collectVolumes(folderLocation, fileName, subFolder)
    print(f"Found {len(volumes)} files.")

    print("Creating histogram and performing normality test...")
    plotHistogramAndTest(volumes, fileName)

# Example usage
if __name__ == "__main__":

    # Set data type
    dataType = 'Test'
    # Set path to the folder containing the data
    if dataType == 'Train':
        folder_location = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/shareData_MRI-only from DATE')
    if dataType == 'Test':
        folder_location = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/shareData_MRI-only DATERANGE test data')
    
    sub_folder = 'MR_StorT2'  # Specify the subfolder name here
    file_names = ['mask_Genitalia.nii.gz', 'mask_Bladder.nii.gz', 'mask_PenileBulb.nii.gz', 'mask_CTVT_427.nii.gz', 'mask_FemoralHead_R.nii.gz', 'mask_PTVT_427.nii.gz', 'mask_FemoralHead_L.nii.gz', 'mask_BODY.nii.gz', 'mask_Rectum.nii.gz']
        
    # Loop through each file name and create histogram for each
    for file_name in file_names:
        main(folder_location, file_name, sub_folder)