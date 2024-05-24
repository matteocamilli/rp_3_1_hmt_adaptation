import os
import re
import pandas as pd
from natsort import natsorted

def extract_cpu_time(file_path):
    """
    Extracts the first occurrence of '-- CPU user time used' from the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r'-- CPU user time used : (\d+) ms', line)
            if match:
                return int(match.group(1))
    return None

def process_files_in_directory(directory_path):
    """
    Processes all text files in the specified directory and extracts the CPU user time used.
    """
    data = []
    filenames = os.listdir(directory_path)
    filenames = natsorted(filenames)


    for filename in filenames:
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            cpu_time = extract_cpu_time(file_path)
            if cpu_time is not None:
                data.append({'Filename': filename, 'CPU User Time Used (ms)': cpu_time})
    
    return pd.DataFrame(data)

# Specify the directory containing the text files
directory_path = "uppaal_logs_renamed/uppaal_logs/RP_round2"

# Process the files and create a DataFrame
df = process_files_in_directory(directory_path)

df1 = pd.read_csv("additional_datasets/initial_configuration_to_improve_random.csv")
df['TAU (ms)'] = df1['PSCS__TAU'] * 1000

# Perform the comparison
count = (df['CPU User Time Used (ms)'] + 7000 < df['TAU (ms)']).sum()
print(count)
print(len(df))

# Save the DataFrame to a CSV file
output_csv_path = "comparison_with_log.csv"
df.to_csv(output_csv_path, index=False)


