#this script splits annotated data with noise into training, validation, and test sets
#it also ensures that noise-free versions of training and validation data are created
#test data is divided into annotated and unannotated files for later evaluation

from pathlib import Path #for handling file operations
import random #for shuffling file lists
import shutil #for copying files
import pandas as pd #for data manipulation
import re #for extracting sample ID

################################################################
#define directories

#source directories
annotated_with_noise_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/annotated_data_with_noise") #where annotated data (with noise) is stored
annotated_no_noise_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/annotated_data") #where annotated data (without noise) is stored
unlabelled_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_preprocessed_data") #where unlabelled test data is stored

#output directories
output_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data") #where split data will be saved
train_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Training_data") #training data directory
train_no_noise_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Training_data_no_noise") #noise-free training data directory
validation_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Validation_data") #validation data directory
validation_no_noise_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Validation_data_no_noise") #noise-free validation data directory
test_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Test_data") #test data directory (unlabelled)
test_annotated_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Test_data_annotated") #test data directory (annotated)

#split ratios
train_ratio = 0.7 #70% of data used for training
work_ratio = 0.15 #15% for validation, leaving 15% for test

output_dirs = [train_dir, train_no_noise_dir, validation_dir, validation_no_noise_dir, test_dir, test_annotated_dir] #output directories

################################################################
#function to clear old files from directories for every code execution

def clear_directory(folder): #removes all existing files in a given directory
    if folder.exists(): #checks if the folder exists before clearing
        for file in folder.iterdir(): #iterates through all files
            if file.is_file(): #ensures only files are deleted, not subdirectories
                file.unlink() #removes the file

################################################################
#create subdirectories and clear old files

for folder in output_dirs: #iterates through each output directory
    folder.mkdir(parents=True, exist_ok=True) #creates directory if it doesn't exist
    clear_directory(folder) #clears any old files in the directory

################################################################
#function to extract sample ID (Axx Sxx Nxx) from filename

def extract_sample_id(filename): #extracts the sample ID from a filename
    match = re.search(r'A\d{2} S\d{2} N\d{2}', filename) #searches for pattern in filename
    return match.group(0) if match else None #returns matched sample ID or None

################################################################
#retrieve and shuffle annotated files (with noise)

annotated_with_noise_files = list(annotated_with_noise_dir.glob("*.csv")) #list all CSV files in annotated data folder
random.shuffle(annotated_with_noise_files) #randomly shuffle the file list for unbiased splitting

#map sample IDs to filenames (with noise)
annotated_with_noise_map = {extract_sample_id(f.name): f for f in annotated_with_noise_files if extract_sample_id(f.name)} #dictionary of sample IDs to file paths

#determine file counts for each category
tot_files = len(annotated_with_noise_files) #total number of annotated files
train_nr = int(tot_files * train_ratio) #number of training files
work_nr = int(tot_files * work_ratio) #number of validation files

#split sample IDs
sample_ids = list(annotated_with_noise_map.keys()) #list of sample IDs from annotated data
train_sample_ids = sample_ids[:train_nr] #selects first portion for training
validation_sample_ids = sample_ids[train_nr:train_nr + work_nr] #selects next portion for validation
test_sample_ids = sample_ids[train_nr + work_nr:] #remaining portion for test data

################################################################
#copy annotated files (with noise) to training, validation, and test sets

for sample_id in train_sample_ids: #loop through training sample IDs
    shutil.copy(annotated_with_noise_map[sample_id], train_dir / annotated_with_noise_map[sample_id].name) #copy file to training folder

for sample_id in validation_sample_ids: #loop through validation sample IDs
    shutil.copy(annotated_with_noise_map[sample_id], validation_dir / annotated_with_noise_map[sample_id].name) #copy file to validation folder

for sample_id in test_sample_ids: #loop through test sample IDs
    shutil.copy(annotated_with_noise_map[sample_id], test_annotated_dir / annotated_with_noise_map[sample_id].name) #copy file to test folder (annotated)

################################################################
#retrieve noise-free data for training and validation

annotated_no_noise_files = {extract_sample_id(f.name): f for f in annotated_no_noise_dir.glob("*.csv") if extract_sample_id(f.name)} #map noise-free annotated files by sample ID

for sample_id in train_sample_ids: #loop through training sample IDs
    if sample_id in annotated_no_noise_files: #check if noise-free version exists
        df = pd.read_csv(annotated_no_noise_files[sample_id]) #load annotated data
        df_no_noise = df[df["Cluster"] != -1] #remove noise rows
        df_no_noise.to_csv(train_no_noise_dir / annotated_no_noise_files[sample_id].name, index=False) #save cleaned data

for sample_id in validation_sample_ids: #loop through validation sample IDs
    if sample_id in annotated_no_noise_files: #check if noise-free version exists
        df = pd.read_csv(annotated_no_noise_files[sample_id]) #load annotated data
        df_no_noise = df[df["Cluster"] != -1] #remove noise rows
        df_no_noise.to_csv(validation_no_noise_dir / annotated_no_noise_files[sample_id].name, index=False) #save cleaned data

################################################################
#find corresponding unannotated test files

unlabelled_files = {extract_sample_id(f.name): f for f in unlabelled_dir.glob("*.csv") if extract_sample_id(f.name)} #map unlabelled test files by sample ID

test_files = [] #empty list to store matched test file names
for sample_id in test_sample_ids: #loop through test sample IDs
    if sample_id in unlabelled_files: #check if unlabelled version exists
        shutil.copy(unlabelled_files[sample_id], test_dir / unlabelled_files[sample_id].name) #copy file to test folder (unlabelled)
        test_files.append(unlabelled_files[sample_id].name) #store file name for reference

################################################################
#print completion message

print("\033[1;32m  Data has been split successfully! \033[0m") #print success message
print(f"( split directories can be found under {output_dir}) ") #print location of split files
