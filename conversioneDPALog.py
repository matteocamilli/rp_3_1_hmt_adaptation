import os
import zipfile
import shutil

zip_path = r'C:\Users\diego\OneDrive\Documents\GitHub\rp_3_1_hmt_adaptation\uppaal_logs.zip'
temp_dir = 'temp_extracted_files'
target_dir_within_zip = 'uppaal_logs/RP_round2'

# Step 1: Extract the specific directory within the ZIP file to a temporary directory
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    members = [m for m in zip_ref.namelist() if m.startswith(target_dir_within_zip)]
    zip_ref.extractall(temp_dir, members)

# Step 2: Rename the .txt files in the extracted directory
extracted_dir = os.path.join(temp_dir, target_dir_within_zip)
for root, _, files in os.walk(extracted_dir):
    for filename in files:
        if filename.endswith(".txt"):
            new_filename = filename.replace(":", "_")
            os.rename(os.path.join(root, filename), os.path.join(root, new_filename))

# Step 3: Create a new ZIP file with the renamed files
new_zip_path = r'C:\Users\diego\OneDrive\Documents\GitHub\rp_3_1_hmt_adaptation\uppaal_logs_renamed.zip'
with zipfile.ZipFile(new_zip_path, 'w') as zip_ref:
    for root, _, files in os.walk(temp_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            arcname = os.path.relpath(file_path, temp_dir)
            zip_ref.write(file_path, arcname)

# Step 4: Clean up the temporary directory
shutil.rmtree(temp_dir)

print(f"Renamed files and created new zip archive at {new_zip_path}")
