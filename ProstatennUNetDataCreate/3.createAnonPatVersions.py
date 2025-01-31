# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physcist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Create unique copies of patiens data for anonymization.
# This is needed as each patient will be created as a unique instance for each observer.
# *********************************************************************************
import os
import pydicom
import csv
from pydicom.uid import generate_uid


def rename_folders(output_root, observer_names, patient_mapping):
    """
    Rename output patient folders using the mapping dictionary and observer name.

    Args:
        output_root (str): Root folder for output anonymized data.
        observer_names (list): List of observer names.
        patient_mapping (dict): Mapping of folder names to patient IDs and names.
    """
    for observer_name in observer_names:
        observer_folder = os.path.join(output_root, 'MRI-only_test_data_' + observer_name, 'inference_DICOM_in')
        for patient_folder in os.listdir(observer_folder):
            old_patient_folder = os.path.join(observer_folder, patient_folder)
            old_patient_id = os.path.basename(patient_folder)
            assert old_patient_id in patient_mapping, f"Folder '{old_patient_id}' not found in patient mapping."
            new_patient_id = patient_mapping[old_patient_id]['newPatientName']
            new_patient_folder = os.path.join(observer_folder, new_patient_id + '_' + observer_name)
            os.rename(old_patient_folder, new_patient_folder)
            print(f"Renamed: {old_patient_folder} -> {new_patient_folder}")


def anonymize_dicom_series(series_folder, output_folder, observer_name, patient_mapping, new_study_uid, new_series_uid):
    """
    Anonymize DICOM series and create unique values for UIDs. Save anonymized DICOM files to the output folder.

    Args:
        series_folder (str): Path to the input DICOM series folder.
        output_folder (str): Path to the output folder for anonymized DICOM series.
        observer_name (str): Name of the observer.
        patient_mapping (dict): Mapping of folder names to new patient IDs and name (same).
        new_study_uid (str): New StudyInstanceUID to be assigned to all DICOM files in the series.
        new_series_uid (str): New SeriesInstanceUID to be assigned to all DICOM files in the series.
    """
    # Loop through all DICOM files in the series folder
    for root, dirs, files in os.walk(series_folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            if file_path.endswith('.dcm'):
                ds = pydicom.dcmread(file_path)
                # Anonymize DICOM metadata
                ds.SeriesInstanceUID = new_series_uid
                ds.SOPInstanceUID = generate_uid()  # Generate a new UID for instance (important)
                ds.StudyInstanceUID = new_study_uid
                # Replace patient ID, name, and keep private tags
                folder_name = os.path.basename(os.path.dirname(series_folder))
                new_patient_info = patient_mapping[folder_name]
                new_patient_id = f"{new_patient_info['newPatientName']}_{observer_name}"
                # Set ID and Name
                ds.PatientID = new_patient_id
                ds.PatientName = new_patient_id
                # Save anonymized DICOM file
                anonymized_file_path = os.path.join(output_folder, filename)
                ds.save_as(anonymized_file_path, write_like_original=True)
                #print(f"Anonymized: {file_path} -> {anonymized_file_path}")


def load_patient_mapping(csv_file):
    """
    Load patient mappings from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing patient mappings.

    Returns:
        dict: Dictionary mapping folder names to patient IDs and names.
    """
    patient_mappings = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            folder_name = row['folderName']
            patient_id = row['realPatientID']
            new_patient_name = row['newPatientName']
            patient_mappings[folder_name] = {'realPatientID': patient_id, 'newPatientName': new_patient_name}
    return patient_mappings


def anonymize_patient_data(input_folder, output_root, observer_name, patient_mapping):
    """
    Anonymize patient data. input_folder is a folder with multiple patient folders, each containing 1 series folder.

    Args:
        input_folder (str): Path to the input DICOM data folder.
        output_root (str): Path to the root folder for output anonymized data.
        observer_names (list): List of observer names.
        patient_mapping (dict): Mapping of folder names to patient IDs and name (the same).
    """
    # Loop through each observer and process the patient data
    observer_output_folder = os.path.join(output_root, 'MRI-only_test_data_' + observer_name, 'inference_DICOM_in')
    os.makedirs(observer_output_folder, exist_ok=True)
    # Process each patient folder
    for patient_folder in os.listdir(input_folder):
        patient_path = os.path.join(input_folder, patient_folder)
        observer_patient_folder = os.path.join(observer_output_folder, patient_folder)
        os.makedirs(observer_patient_folder, exist_ok=True)
        # Get all series folders within the patient folder
        series_folders = [f.path for f in os.scandir(patient_path) if f.is_dir()]
        # Select only folder paths with name 'MR_StorT2'
        series_folders = [f for f in series_folders if os.path.basename(f) == 'MR_StorT2']
        # Generate StudyInstanceUID and SeriesInstanceUID for this patient folder
        # Needs to be the same within a series folder
        new_study_uid = generate_uid()
        new_series_uid = generate_uid()

        # Process each series folder within the patient folder
        for series_folder in series_folders:
            series_name = os.path.basename(series_folder)
            observer_series_folder = os.path.join(observer_patient_folder, series_name)
            os.makedirs(observer_series_folder, exist_ok=True)

            # Anonymize DICOM series within the series folder
            anonymize_dicom_series(series_folder, observer_series_folder, observer_name, patient_mapping,
                                    new_study_uid, new_series_uid)
            print(f"Anonymized: {series_folder} -> {observer_series_folder}")


# Script entry point
if __name__ == '__main__':
    # Example usage
    input_folder = "/mnt/mdstore2/Christian/MRIOnlyData/inference/MRI-only_test_data/inference_DICOM_in/"
    output_root = "/mnt/mdstore2/Christian/MRIOnlyData/inference/newA/"
    observer_names = ['obsA', 'obsB', 'obsC', 'obsD', 'obsE', 'obsF']
    #observer_names = ['obsA']
    #observer_names = ['obsA']
    csv_file = "/mnt/mdstore2/Christian/MRIOnlyData/patient_ID_mappings.csv"

    # Load patient mappings from CSV file
    patient_mappings = load_patient_mapping(csv_file)

    # Anonymize patient data for each observer
    for observer_name in observer_names:
        anonymize_patient_data(input_folder, output_root, observer_name, patient_mappings)

    # Rename output folders to the new patient IDs
    rename_folders(output_root, observer_names, patient_mappings)
