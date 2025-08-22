import os
import csv
import pydicom

def is_dicom_file(file_path):
    """Check if the file is a valid DICOM file"""
    try:
        pydicom.dcmread(file_path, force=True)
        return True
    except:
        return False

def find_dicom_files(root_dir):
    """Find all DICOM files in the directory and subdirectories that meet both conditions"""
    dicom_files = []
    required_prefix = r"D:\dicom-compressors\metadata\4D-Lung"

    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.lower().endswith(".dcm") and file_path.startswith(required_prefix):
                if is_dicom_file(file_path):
                    dicom_files.append(file_path)
    
    return dicom_files

def save_to_csv(file_paths, output_file):
    """Save the file paths to a CSV file"""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['DICOM File Path'])
        writer.writerows([[path] for path in file_paths])

if __name__ == "__main__":
    # Fixed directory path
    search_directory = r"D:\dicom-compressors"
    
    # Output CSV filename
    output_csv = "dicom_paths.csv"
    
    print(f"Searching for DICOM files in {search_directory}...")
    dicom_files = find_dicom_files(search_directory)
    
    if dicom_files:
        save_to_csv(dicom_files, output_csv)
        print(f"Saved {len(dicom_files)} DICOM file paths to {output_csv}")
    else:
        print("No DICOM files found in the specified directory.")
