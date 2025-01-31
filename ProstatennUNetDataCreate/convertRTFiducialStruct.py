# *********************************************************************************
# Author: Christian Jamtheim Gustafsson, PhD, Medical Physicist Expert
# Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
# Description: Convert RT marker structure to Nifti format for public repository
# Also output the marker DICOM coordinates to a text file. 
# *********************************************************************************
import os
import subprocess
import nibabel as nib
import numpy as np
import re
import numpy as np
from scipy.ndimage import label

# Configuration block for the C3D software path
#https://sourceforge.net/projects/c3d/files/c3d/Experimental/
#https://sourceforge.net/p/c3d/git/ci/master/tree/doc/c3d.md
CONFIG = {
    "C3D": "/mnt/mdstore2/Christian/YngreAlf/SegUnc/src/software/c3d-1.1.0-Linux-x86_64/bin/c3d", 
}

class RTFiducialProcessor:
    def __init__(self, rtstruct_path, nifti_image_path, output_dir):
        """
        Initializes the processor with the RTSTRUCT and NIfTI image paths and the output directory.
        
        Args: 
            rtstruct_path: Full path to the RTSTRUCT DICOM file.
            nifti_image_path: Full path to the NIfTI image file.
            output_dir: Directory to store the output NIfTI file and the txt file.
        """
        self.rtstruct_path = rtstruct_path
        self.nifti_image_path = nifti_image_path
        self.output_dir = output_dir


    def convert_rtstruct(self):
        """
        Converts the RTSTRUCT DICOM file to a NIfTI image with landmarks (spheres).
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Extract coordinates from RTSTRUCT file using dcmdump
        dcmdump_output = subprocess.run(["dcmdump", self.rtstruct_path], capture_output=True, text=True)
        if dcmdump_output.returncode != 0:
            raise Exception(f"Error running dcmdump on {self.rtstruct_path}: {dcmdump_output.stderr}")
        
        # Save dcmdump output to a file with patient_number as suffix
        dcmdump_file = os.path.join(self.output_dir, f"dcmdump_RTStruct_tmp.txt")
        with open(dcmdump_file, "w") as f:
            f.write(dcmdump_output.stdout)

        # Extract the ContourData
        # Init a list to store the extracted coordinates
        coords = []
        # Open the dcmdump file and read line by line
        with open(dcmdump_file, "r") as file:
            for line in file:
                # Check if the line contains "ContourData"
                if "ContourData" in line:
                    # Extract the content inside the square brackets, then find all numbers within that content
                    contour_data = list(map(float, re.findall(r'-?\d+\.?\d*', re.search(r'\[(.*?)\]', line).group(1))))
                    # Assert three values are extracted as we expect x, y, z coordinates for each point
                    assert len(contour_data) == 3
                    # Save them in coords as a new line
                    coords.append(f"{contour_data[0]}, {contour_data[1]}, {contour_data[2]}")
            # Assert that three lines are extracted as we expect three markers/points
            # Some patients have only 2 markers, so we need to check for that
            if 'AAA' in self.rtstruct_path or 'BBB' in self.rtstruct_path or 'CCC' in self.rtstruct_path:
                assert len(coords) == 2, f"Expected 2 coordinates, found {len(coords)} for patient {self.rtstruct_path}"
            else: 
                assert len(coords) == 3, f"Expected 3 coordinates, found {len(coords)} for patient {self.rtstruct_path}"
        
        # File can be removed now
        os.remove(dcmdump_file)
        # Print the extracted coordinates
        # print(f"Coordinates extracted: {coords}")


        # Process the coordinates and save coordinates to a text file
        # Define file paths for original and flipped coordinates
        # Flipped coordinates are used by C3D to generate the final NIfTI file (Nifti coordinates are flipped)
        # Original coordinates are saved for reference
        coords_file_path = os.path.join(self.output_dir, f"MRI_T2_DICOM_coords_fiducials.txt")
        flipped_coords_file_path = os.path.join(self.output_dir, f"MRI_T2_DICOM_coords_fiducials_flipped.txt")
        # Set value in segmentation to 1
        value = 1
        # Open both files and write coordinates to each
        with open(coords_file_path, "w") as coords_file, open(flipped_coords_file_path, "w") as flipped_coords_file:
            # Write each coordinate set to the files
            for coord_set in coords:
                try:
                    x, y, z = map(float, coord_set.split(","))  # Convert to float and split by comma
                    # Write the original coordinates to the first file
                    coords_file.write(f"{x},{y},{z}\n")
                    # Flip signs for x and y, then write to the second file
                    flipped_x, flipped_y = -x, -y
                    flipped_coords_file.write(f"{flipped_x} {flipped_y} {z} {value}\n")

                except ValueError as e:
                    print(f"Skipping invalid coordinate set: {coord_set} - Error: {e}")


        # Call C3D to generate the final NIfTI with the markers as spheres with radius 3 mm
        output_nifti = os.path.join(self.output_dir, f"mask_MRI_T2_coords_fiducials.nii.gz")
        c3d_command = [
            CONFIG["C3D"], self.nifti_image_path, "-scale", "0",
            "-landmarks-to-spheres", flipped_coords_file_path, "3", "-o", output_nifti
        ]
        c3d_output = subprocess.run(c3d_command, capture_output=True, text=True)

        if c3d_output.returncode != 0:
            raise Exception(f"Error running c3d on {self.nifti_image_path}: {c3d_output.stderr}")
        else: 
            # Remove the flipped coordinates file
            os.remove(flipped_coords_file_path)

        print(f"Output file created: {output_nifti}")

        # Validate the final NIfTI file
        self.validate_nifti(output_nifti)


    def validate_nifti(self, nifti_path):
        """
        Validates the generated NIfTI file by checking that the maximum value is 1.
        
        Args:
            nifti_path: Full path to the output NIfTI file.
        """
        # Load the NIfTI file using nibabel
        img = nib.load(nifti_path)
        data = img.get_fdata()
        # Check that the maximum value is 1 
        max_val = np.max(data)
        # Assert max value is 1
        assert max_val == 1, f"Validation failed: max value is {max_val}."
        # Check how many disconnected objects are in the array
        # Label connected components in the array
        labeled_array, num_features = label(data)
        # Check if there are exactly 3 disconnected spheres, otherwise raise an error
        if 'X1' in self.rtstruct_path or 'X2' in self.rtstruct_path or 'X3' in self.rtstruct_path or 'X4' in self.rtstruct_path or 'X5' in self.rtstruct_path:  # First three patient has 2, rest has 3 with overlap. 
            if num_features != 2:
                raise Exception(f"Validation failed: expected 2 disconnected spheres, found {num_features}. For patient {self.rtstruct_path}.")
        else: 
            if num_features != 3:
                raise Exception(f"Validation failed: expected 3 disconnected spheres, found {num_features}. For patient {self.rtstruct_path}.")
            
