# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physicist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Visual assessment of image volumes in Nifti files using ITK-SNAP.
# This script loops through MRI and CT image volumes for each patient, opens them
# in ITK-SNAP, and logs the processed patients in a log file. MRI is processed first,
# followed by CT and registered CT volume. The program checks the log file to avoid
# re-processing.
# *********************************************************************************

import os
import subprocess
import time

LOG_FILE = "NIFTI_QA_processed_patients_sharedData.log"
# Define location of the log file as the same folder as the script
LOG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), LOG_FILE)


def log_patient(patient_id):
    """
    Log the patient ID to the log file.

    Args:
        patient_id (str): Patient identifier to log.
    """
    with open(LOG_FILE, "a") as log:
        log.write(f"{patient_id}\n")


def is_patient_logged(patient_id):
    """
    Check if a patient ID is already logged.

    Args:
        patient_id (str): Patient identifier to check.

    Returns:
        bool: True if the patient ID is in the log file, False otherwise.
    """
    if not os.path.exists(LOG_FILE):
        return False
    with open(LOG_FILE, "r") as log:
        logged_patients = log.read().splitlines()
    return patient_id in logged_patients


def load_and_open_file(file_path):
    """
    Load and open a Nifti file in ITK-SNAP. Raise an exception if the file does not exist.

    Args:
        file_path (str): Path to the Nifti file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Opening {file_path} in ITK-SNAP...")
    subprocess.run(["itksnap", "-g", file_path])



def process_patient(patient_folder):
    """
    Process MRI and CT data for a single patient.

    Args:
        patient_folder (str): Path to the patient folder containing MRI and CT subfolders.
    """
    patient_id = os.path.basename(patient_folder)

    if is_patient_logged(patient_id):
        print(f"Patient {patient_id} already processed. Skipping.")
        return

    try:
        # Paths to MRI and CT data
        mri_path = os.path.join(patient_folder, "MR_StorT2", "image.nii.gz")
        ct_path = os.path.join(patient_folder, "sCT", "image.nii.gz")
        ct_reg_path = os.path.join(patient_folder, "sCT", "image_reg2MRI.nii.gz")

        # Process MRI first
        load_and_open_file(mri_path)
        # Process registered CT
        load_and_open_file(ct_reg_path)
        # Process CT next
        #load_and_open_file(ct_path)
        

        # Log the patient after successful processing
        log_patient(patient_id)
        print(f"Patient {patient_id} processed successfully.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Skipping patient {patient_id} due to missing data.")


def main(folder_location):
    """
    Main function to process all patient folders.

    Args:
        folder_location (str): Path to the root folder containing patient data.
    """
    print(f"Scanning folder: {folder_location}")

    for patient_folder in sorted(os.listdir(folder_location)):
        full_path = os.path.join(folder_location, patient_folder)
        if os.path.isdir(full_path):
            process_patient(full_path)


# Example usage
if __name__ == "__main__":

    # Set data type
    dataType = 'Train'
    # Set path to the folder containing the data
    if dataType == 'Train':
        folder_location = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/shareData_MRI-only from DATE')
    if dataType == 'Test':
        folder_location = os.path.join('/mnt/mdstore2/Christian/MRIOnlyData/shareData_MRI-only DATERANGE test data')

    main(folder_location)
