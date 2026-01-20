import os
import zipfile
import shutil
import gzip
import pandas as pd

def extract_mimic_demo(zip_path="mimic-iv-clinical-database-demo-2.2.zip", output_dir='data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extract_path = "mimic-iv-demo-extracted"

    if not os.path.exists(zip_path):
        print(f"Error: Zip file '{zip_path}' not found.")
        return

    print(f"Extracting zip file: {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction complete.")
    except Exception as e:
        print(f"Failed to extract: {e}")
        return

    # Find and move relevant files
    # We need: patients.csv, admissions.csv, diagnoses_icd.csv, labevents.csv
    # They might be .csv.gz
    
    target_basenames = ['patients', 'admissions', 'diagnoses_icd', 'labevents']
    
    found_files = {}
    
    print("Scanning for files...")
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            # Check if file matches any target basename (with .csv or .csv.gz)
            for base in target_basenames:
                if file == f"{base}.csv" or file == f"{base}.csv.gz":
                    found_files[base] = os.path.join(root, file)
                    
    print(f"Found files: {found_files}")
    
    for base, src_path in found_files.items():
        dst_path = os.path.join(output_dir, f"{base}.csv")
        
        if src_path.endswith('.gz'):
            print(f"Decompressing {src_path} to {dst_path}...")
            with gzip.open(src_path, 'rb') as f_in:
                with open(dst_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print(f"Moving {src_path} to {dst_path}...")
            shutil.copy(src_path, dst_path)

    # Cleanup
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
        
    print("Data setup complete.")

if __name__ == "__main__":
    extract_mimic_demo()
